#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# New modules to import
import argparse
import pathlib

# Modules imported from original .ipynb file
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from os.path import join as pathjoin
import h5py
import time

import imp
import sys
sys.path.append('..')
import donglab_workflows as dw
imp.reload(dw)
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import tifffile as tf # for 16 bit tiff
import h5py

parser = argparse.ArgumentParser(description="Downsample the lightsheet")

parser.add_argument("input_path",type=pathlib.Path,help="The input path, can be a directory or a filename")
parser.add_argument("image_type",choices=["ims","tif"],help="The image type, can be either ims or tif")
parser.add_argument("-of","--output_filename",default=None,help="The name of the generated output file")
parser.add_argument("outdir",type=pathlib.Path,help="The temporary output directory for intermediate results")
parser.add_argument("-dI",default=None,help="Deviation Index")
parser.add_argument("-res",default=50.0,type=np.float32,help="Desired voxel size")
parser.add_argument("-c","--channel",default=0,type=int,help="Specify channel number")
parser.add_argument("-dss","--dataset_string",default=None)
parser.add_argument("-cs","--chunksize",default=None,help="chunksize for looking for areas with no data and loading quickly")
parser.add_argument("-bs","--blocksize",default=None,help="blocksizesize for looking for areas with no data and loading quickly")

args = parser.parse_args()

input_path      = args.input_path
image_type      = args.image_type
output_filename = args.output_filename
outdir          = args.outdir
dI              = args.dI
res             = args.res
channel         = args.channel
dataset_string  = args.dataset_string
chunksize       = args.chunksize
blocksize       = args.blocksize

# power to reduce dynamic range
power = np.ones(1,dtype=np.float32)*0.125

# blocksize and chunksize for looking for areas with no data and loading quickly

# build a tif class with similar interface
class TifStack:
    '''We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    We assume 16
    '''
    def __init__(self,input_directory,pattern='*.tif'):
        self.input_directory = input_directory
        self.pattern = pattern
        self.files = glob(pathjoin(input_directory,pattern))
        self.files.sort()
        test = Image.open(self.files[0])
        self.nxy = test.size
        test.close()
        self.nz = len(self.files)
        self.shape = (self.nz,self.nxy[1],self.nxy[0]) # note, it is xy not rowcol
    def __getitem__(self,i):
        return tf.imread(self.files[i])/(2**16-1)
    def __len__(self):
        return len(self.files)
    def close(self):
        pass # nothing necessary

# if none will load from data
if dI is None and image_type == 'ims':
    f = h5py.File(input_path,'r')
    dI = dw.imaris_get_pixel_size(f)    
    xI = dw.imaris_get_x(f)
    f.close()

if output_filename is None:
    output_filename = os.path.splitext(os.path.split(input_path)[-1])[0] + '_ch_' + str(channel) + '_pow_' + str(power) + '_down.npz'

if dataset_string is None and image_type == 'ims':
    dataset_string = f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data' # not used for Tifs

if image_type == 'tif':
    data = TifStack(input_path)
    
elif image_type == 'ims':
    data_ = h5py.File(input_path,mode='r')
    data = data_[dataset_string]

# get the data
if chunksize is None:
    chunksize = data.chunks[0]
if blocksize is None:
    blocksize = data.chunks[1]

print(f'Input path is {input_path}')
print(f'Output filename is {output_filename}')
print(f'Resolution is {dI}')
print(f'Desired resolution is {res}')
print(f'Dataset string is {dataset_string}')
print(f'Temp output dir is {outdir}')

down = np.floor(res/dI).astype(int)

nI = np.array(data.shape)
if not(image_type == 'ims'):
    # if we couldn't calculate xI above, we'lluse these defaults
    xI = [np.arange(n)*d - (n-1)/2.0*d for n,d in zip(nI,dI)]
nIreal = np.array([len(x) for x in xI])

xId = [dw.downsample(x,[d]) for x,d in zip(xI,down)]
dId = [x[1]-x[0] for x in xId]

# Iterate over the dataset (currently not doing weights) + Save intermediate outputs 
# (each slice) in case of errors
fig,ax = plt.subplots(2,2)
ax = ax.ravel()
working = []
working2 = []
workingw = []
output = []
output2 = []
outputw = []
start = time.time()

for i in range(nIreal[0]):
    starti = time.time()
    outname = os.path.join(outdir,f'{i:06d}_s.npy')

    ##########
    # Note from Daniel on January 23, 2023
    # s2 is for measuring local variance, as an additional feature that preserves some high resolution information
    # in the future we may want to disable this, since it doubles the compute time (although it does not affect the network/io time)
    ##########
    outnames2 = outname.replace('_s','_s2')
    outnamew = outname.replace('_s','_w')
    if os.path.exists(outname) and os.path.exists(outnames2) and os.path.exists(outnamew):
        # what happens if it fails in the middle of a chunk?
        sd = np.load(outname)
        s2d = np.load(outnames2)
        wd = np.load(outnamew)
    else:
        # load a whole chunk
        if not i%chunksize:
            data_chunk = data[i:i+chunksize]
        # use this for weights
        #s_all = data[i,:,:]
        # it's possible that this will fail if I haven't defined data_chunk yet
        try:
            s_all = data_chunk[i%chunksize,:,:]
        except:
            # we need to load, not starting at i
            # but at the beginning of the chunk
            data_chunk = data[i//chunksize:i//chunksize+chunksize]
            s_all = data_chunk[i%chunksize,:,:]
        s = s_all[:nIreal[1]+1,:nIreal[2]+1]**power # test reduce dynamic range before downsampling with this power
        s2 = s**2
        #w = (s>0).astype(float)
        # this is not a good way to get weights, 
        # we need to look for a 64x64 block of all zeros
        
        s_all_block = s_all.reshape(s_all.shape[0]//blocksize,blocksize,s_all.shape[1]//blocksize,blocksize)
        tmp = np.logical_not(np.all(s_all_block==0,axis=(1,3))).astype(np.uint8)
        s_all_w = np.ones_like(s_all_block)
        s_all_w *= tmp[:,None,:,None]
        s_all_w = s_all_w.reshape(s_all.shape)
        w = s_all_w[:nIreal[1]+1,:nIreal[2]+1].astype(power.dtype)

        
        sd = dw.downsample((s*w),down[1:])
        s2d = dw.downsample((s2*w),down[1:])
        wd = dw.downsample(w,down[1:])
        sd /= wd
        sd[np.isnan(sd)] = 0.0
        s2d /= wd
        s2d[np.isnan(s2d)] = 0.0
    
        np.save(outname,sd)
        np.save(outname.replace('_s','_w'),wd)
        np.save(outname.replace('_s','_s2'),s2d)
    
    ax[0].cla()
    wd0 = wd>0.0
    if np.any(wd0):
        vmin = np.min(sd[wd0])
        vmax = np.max(sd[wd0])
    else:
        vmin = None
        vmax = None
    ax[0].cla()
    ax[0].imshow(sd,vmin=vmin,vmax=vmax)
    ax[2].cla()
    ax[2].imshow(wd,vmin=0,vmax=1)
    working.append(sd)
    working2.append(s2d)
    workingw.append(wd)
    
    if len(working) == down[0]:
        workingw_stack = np.stack(workingw)
        out = dw.downsample(np.stack(working)*workingw_stack,[down[0],1,1])
        out2 = dw.downsample(np.stack(working2)*workingw_stack,[down[0],1,1])
        outw = dw.downsample(workingw_stack,[down[0],1,1])        
        out /= outw
        out[np.isnan(out)] = 0.0
        out2 /= outw
        out2[np.isnan(out2)] = 0.0
        outstd = out2 - out**2
        outstd[outstd<0]=0
        outstd = np.sqrt(outstd)
        wd0 = (wd>0.0)[None]
        if np.any(wd0):
            outshow = (out[0] - np.min(out[wd0]))/(np.quantile(out[wd0],0.99) - np.min(out[wd0]))
            outshowstd = (outstd[0] - np.min(outstd[wd0]))/(np.quantile(outstd[wd0],0.99) - np.min(outstd[wd0]))
        else:
            outshow = (out[0] - np.min(out))/(np.quantile(out,0.99) - np.min(out))
            outshowstd = (outstd[0] - np.min(outstd))/(np.quantile(outstd,0.99) - np.min(outstd))
        ax[1].cla()
        ax[1].imshow(np.stack((outshow,outshowstd,outshow),-1))
        ax[3].cla()
        ax[3].imshow(outw[0],vmin=0,vmax=1)
        output.append(out)
        output2.append(out2)
        outputw.append(outw)
        working = []
        workingw = []
        working2 = []
    fig.suptitle(f'slice {i} of {data.shape[0]}')
    fig.canvas.draw()
    print(f'Finished loading slice {i} of {data.shape[0]}, time {time.time() - starti} s')

output = np.concatenate(output)        
Id = output
wd = np.concatenate(outputw)
print(f'Finished downsampling, time {time.time() - start}')

np.savez(output_filename,I=Id,I2=np.concatenate(output2),xI=np.array(xId,dtype='object'),w=wd) # note specify object to avoid "ragged" warning

fig,ax = dw.draw_slices(Id,xId)
fig.suptitle(output_filename)
fig.savefig(output_filename.replace('npz','jpg'))
