#!/usr/bin/env python
# coding: utf-8

# New modules to import
import argparse
import pathlib

# Modules imported from original .ipynb file
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import csv
import pandas as pd
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# for now add emlddmm library for registration
sys.path.append('/home/dtward/data/csh_data/emlddmm')
import emlddmm

parser = argparse.ArgumentParser(description="Register images in atlas")

parser.add_argument("target_name",type=pathlib.Path,help="Name of the file to be registered")
parser.add_argument("output_prefix",type=pathlib.Path,help="Location of the output file(s)")
parser.add_argument("seg_name",type=pathlib.Path,help="Location of segmentation file")
parser.add_argument("savename",type=pathlib.Path,help="Name of file once it is saved")
parser.add_argument("atlas_names",type=pathlib.Path,nargs='+',help="Location of the atlas images")
parser.add_argument("-d","--device",default='cuda:0',help="Device used for pytorch")
parser.add_argument("-a","--A0",type=str,default=None,help="Affine transformation (Squeezed to 16x1 + Sep by ',')")
parser.add_argument("-res","--resolution", type=np.float32, choices=[20.0, 50.0], default=20, help="Resoultion used during downsampling")
parser.add_argument("-j","--jpg", action="store_true", help="Generate 3d jpegs for each structure")

args = parser.parse_args()

target_name   = args.target_name
output_prefix = args.output_prefix
atlas_names   = args.atlas_names
seg_name      = args.seg_name
savename      = args.savename
resolution    = args.resolution
jpg           = args.jpg
device        = args.device
A0            = args.A0


##########
# Note from Daniel on January 23, 2023 
# check if specified device is available
import torch
if 'cuda' in device:
    # you may still get into trouble if cuda:1 is selected but only cuda:0 is available
    # we should update this to check more carefully, but this is probably okay for now
    assert(torch.cuda.is_available(), 'You selected device cuda but it is not available. exiting')
##########

# Specify output directory
output_directory = os.path.split(output_prefix)[0]
if output_directory:
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

# Load the target image
target_data = np.load(target_name,allow_pickle=True)

J = target_data['I'][None]
J = J.astype(np.float32)
J /= np.mean(np.abs(J))
xJ = target_data['xI']
dJ = [x[1] - x[0] for x in xJ]
J0 = np.copy(J)
origin = np.array([xJ[0][0], xJ[1][0], xJ[2][0]])  # Assumming a x, y, z origin

if 'w' in target_data:
    W = target_data['w']
elif 'W' in target_data:
    W = target_data['W']
else:
    W = np.ones_like(J[0])
    # or actually
    W = (J[0]>0).astype(float)
    
W = (J[0]>0).astype(float)

emlddmm.draw(W[None])

fig,ax = emlddmm.draw(J,xJ,vmin=np.min(J[W[None]>0.9]))
fig.suptitle('Downsampled lightsheet data')
figopts = {'dpi':300}
fig.savefig(os.path.join(output_prefix, 'downsampled.jpg'),**figopts)
fig.canvas.draw()

I = []
for atlas_name in atlas_names:
    xI,I_,title,names = emlddmm.read_data(atlas_name)
    I_ = I_.astype(np.float32)
    I_ /= np.mean(np.abs(I_))
    I.append(I_)

I = np.concatenate(I)
dI = [x[1] - x[0] for x in xI]
XI = np.meshgrid(*xI,indexing='ij')

xI0 = [ np.copy(x)for x in xI]
I0 = np.copy(I)

fig,ax = emlddmm.draw(I,xI,vmin=0)
fig.canvas.draw()

# next is to transforom the high resolution data
xS,S,title,names = emlddmm.read_data(seg_name)
# we want to visuze the above with S
labels,inds = np.unique(S,return_inverse=True)

colors = np.random.rand(len(labels),3)
colors[0] = 0.0

RGB = colors[inds].reshape(S.shape[1],S.shape[2],S.shape[3],3).transpose(-1,0,1,2)
#RGB = np.zeros((3,S.shape[1],S.shape[2],S.shape[3]))

#for i,l in enumerate(labels):
#    RGB[:,S[0]==l] = colors[i][...,None]
fig,ax = emlddmm.draw(RGB)
plt.subplots_adjust(wspace=0,hspace=0,right=1)
fig.canvas.draw()


### Initial Preprocessing
# Target preprocessing

# background
J = J0 - np.quantile(J0[W[None]>0.9],0.1)
J[J<0] = 0
# adjust dynamic range
J = J**0.25

# adjust mean value
J /= np.mean(np.abs(J))
fig,ax = emlddmm.draw(J,xJ,vmin=0)
fig.canvas.draw()
fig.suptitle('Preprocessed lightsheet data')
fig.savefig(os.path.join(output_prefix, 'processed.jpg'), **figopts)


fig,ax = plt.subplots()
ax.hist(J.ravel(),100,log=True)
fig.canvas.draw()

# Atlas preprocessing

# pad
# since I will downsample by 4, I want to pad with 4x4x4
npad = 4
I = np.pad(I0,  ((0,0),(npad,npad),(npad,npad),(npad,npad)) )
for i in range(npad):
    xI = [ np.concatenate(   (x[0][None]-d,x,x[-1][None]+d)   ) for d,x in zip(dI,xI0)]

# adjust nissl image dynamic range
I[0] = I[0]**0.5
I[0] /= np.mean(np.abs(I[0]))
I[1] /= np.mean(np.abs(I[1]))

fig,ax = emlddmm.draw(I,xI,vmin=0)
fig.canvas.draw()


### Registration

if A0 == None:
    # initial affine
    A0 = np.eye(4)
    # make sure to keep sign of Jacobian
    #A0 = np.diag((1.3,1.3,1.3,1.0))@A0
    # flip x0,x1
    A0 = np.array([[0.0,-1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])@A0
    # flip x0,x2
    A0 = np.array([[0.0,0.0,-1.0,0.0],[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0]])@A0
    # flip x1,x2
    #A0 = np.array([[-1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]])@A0
    # flip
    #A0 = np.diag((1.0,-1.0,-1.0,1.0))@A0
    # shift
    #A0[0,-1] = +3500 # left right, positive will cut off the missing hemisphere as appropriate
    #A0[1,-1] = -2500 # AP, negative will cut off the nose
    #A0[2,-1] = -1000 # SI, negative will move the venttral surface toward the boundary

    XJ = np.meshgrid(*xJ,indexing='ij')
    A0[:3,-1] = np.mean(XJ,axis=(-1,-2,-3))

    # check it
    tform = emlddmm.Transform(A0,direction='b')
    AI = emlddmm.apply_transform_float(xI,I,tform.apply(XJ))
    fig,ax = emlddmm.draw(np.concatenate((AI[:2],J)),xJ,vmin=0)
    fig.canvas.draw()
else:  # Use else here, otherwise this will always run
    A0 = np.fromstring(A0,sep=',').reshape(4,4)

# now we want to register
config0 = {
    'device':device,
    'n_iter':200, 'downI':[8, 8, 8], 'downJ':[8, 8, 8],
    'priors':[0.9,0.05,0.05],'update_priors':False,
    'update_muA':0,'muA':[np.quantile(J,0.99)],
    'update_muB':0,'muB':[0.0],
    'update_sigmaM':0,'update_sigmaA':0,'update_sigmaB':0,
    'sigmaM':0.25,'sigmaB':0.5,'sigmaA':1.25,
    'order':1,'n_draw':50,'n_estep':3,'slice_matching':0,'v_start':1000,
    'eA':5e4,'A':A0,'full_outputs':True,
}

if resolution == 50:
    config0['downI'] = [4, 4, 4]
    config0['downJ'] = [4, 4, 4]

# update my sigmas (august 22)
config0['sigmaM'] = 1.0
config0['sigmaB'] = 2.0
config0['sigmaA'] = 5.0
#W = np.ones_like(J[0])
#W = 1.0 - (J[0]==0)
#I_ = np.stack((I[0],I[1],I[0]**2,I[0]*I[1],I[1]**2,))
#I_ = np.stack((I[2],I[0],I[1],I[0]**2,I[0]*I[1],I[1]**2))
#I_ = np.stack((I[0],I[1],I[0]*I[1],I[0]**2,I[1]**2)) # I don't need I[2]
I_ = np.stack((I[0]-np.mean(I[0]),I[1]-np.mean(I[1]),
               (I[0]-np.mean(I[0]))**2,
               (I[0]-np.mean(I[0]))*(I[1]-np.mean(I[1])),
               (I[1]-np.mean(I[0]))**2,))

out = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config0)

# second run, with deformation
config1 = dict(config0)
config1['A'] = out['A']
config1['eA'] = config0['eA']*0.1
config1['a'] = 1000.0
config1['sigmaR'] = 5e4 # 1e4 gave really good results, but try 2e4, also good, I showed this in my slides
config1['n_iter']= 2000
config1['v_start'] = 0
config1['ev'] = 1e-2
config1['ev'] = 2e-3 # reduce since I decreased sigma
config1['v_res_factor'] = config1['a']/dI[0]/4 # what is the resolution of v, as a multiple of that in I
config1['local_contrast'] = [32,32,32]
config1['local_contrast'] = [16,16,16]

if resolution == 50:
    config1['v_res_factor'] = config1['a']/dI[0]/2

# config1['device'] = 'cpu' # we will keep the same device and not update it, this can cause problems

#I_ = np.stack((I[2],I[0],I[1],I[0]**2,I[0]*I[1],I[1]**2,))

# note this approach has an issue with background being black
#I_ = np.stack((I[2],I[0]-np.mean(I[0]),I[1]-np.mean(I[1]),
#               (I[0]-np.mean(I[0]))**2,
#               (I[0]-np.mean(I[0]))*(I[1]-np.mean(I[1])),
#               (I[1]-np.mean(I[0]))**2,))
I_ = np.stack((I[0]-np.mean(I[0]),I[1]-np.mean(I[1]),
               (I[0]-np.mean(I[0]))**2,
               (I[0]-np.mean(I[0]))*(I[1]-np.mean(I[1])),
               (I[1]-np.mean(I[0]))**2,))


out1 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config1)

# on the next run we do les downsampling
config2 = dict(config1)
config2['A'] = out1['A']
config2['n_iter']= 1000
config2['v'] = out1['v']
config2['downI'] = [4, 4, 4]
config2['downJ'] = [4, 4, 4]

if resolution == 50:
    config2['downJ'] = [2, 2, 2]
    config2['downI'] = [2, 2, 2]

# there seems to be an issue with the initial velocity
# when I run this twice, I'm reusing it
out2 = emlddmm.emlddmm(xI=xI,I=I_,xJ=xJ,J=J, W0=W, **config2)
# the matching energy here is way way lower, why would that be?

# save the outputs
np.save(os.path.join(output_prefix, savename), np.array([out2],dtype=object))

out2['figI'].savefig(os.path.join(output_prefix, 'transformed.jpg'), **figopts)
out2['figfI'].savefig(os.path.join(output_prefix, 'contrast.jpg'), **figopts)
out2['figErr'].savefig(os.path.join(output_prefix, 'err.jpg'), **figopts)

fig = out2['figErr']
axs = fig.get_axes()
for ax in axs:
    ims = ax.get_images()
    for im in ims:
        im.set_cmap('twilight')
        clim = im.get_clim()
        lim = np.max(np.abs(clim))
        im.set_clim(np.array((-1,1))*lim)
fig.canvas.draw()
fig.savefig(os.path.join(output_prefix, 'err.jpg'), **figopts)


### Prepare some visualizations

# compute transform for atlas and labels
deformation = emlddmm.Transform(out2['v'],domain=out2['xv'],direction='b')
affine = emlddmm.Transform(out2['A'],direction='b')
tform = emlddmm.compose_sequence([affine,deformation],XJ)

# transform the atlas and labels, notice different domains
It = emlddmm.apply_transform_float(xI,I,tform).cpu().numpy()
RGBt = emlddmm.apply_transform_float(xS,RGB,tform).cpu().numpy()
St = emlddmm.apply_transform_int(xS,S,tform,double=True,padding_mode='zeros').cpu().numpy()

#fig,ax = emlddmm.draw(np.stack((It[0]*0.5,It[1]*0.5,Jsave[0]*1.5)),xJ,vmin=0,vmax=4)
#fig.subplots_adjust(wspace=0,hspace=0,right=1)
fig,ax = emlddmm.draw(np.stack((It[0]*0.5,It[1]*0.5,J0[0]*1.5)),xJ,)
fig.subplots_adjust(wspace=0,hspace=0,right=1)
fig.savefig(os.path.join(output_prefix, 'IJsave.jpg'))

fig,ax = emlddmm.draw(It,xJ)
fig.subplots_adjust(wspace=0,hspace=0,right=1)
fig.savefig(os.path.join(output_prefix, 'Isave.jpg'))

fig,ax = emlddmm.draw(J,xJ)
plt.subplots_adjust(wspace=0,hspace=0,right=1)
fig.savefig(os.path.join(output_prefix, 'Jsave.jpg'))

# transform the target to atlas
# for visualizatoin, we want to sample at xS so we can view it relative to the
XS = np.stack(np.meshgrid(*xS,indexing='ij'))
deformation = emlddmm.Transform(out2['v'],domain=out2['xv'],direction='f')
affine = emlddmm.Transform(out2['A'],direction='f')
tformi = emlddmm.compose_sequence([deformation,affine,],XS)
#J_ = J0**0.25
J_ = np.copy(J)
Jt = emlddmm.apply_transform_float(xJ,J_,tformi,padding_mode='zeros').cpu().numpy()

# view the transformed target
fig,ax = emlddmm.draw(Jt,xS,vmin=np.quantile(J_,0.02),vmax=np.quantile(J_,0.98))
fig.subplots_adjust(wspace=0,hspace=0,right=1)

# view the transformed target with labels
minval = 1.5
maxval = 2.9
minval = 0.0
maxval = 5.0
alpha = 0.3
alpha = 0.75
fig = plt.figure(figsize=(7,7))
n = 4
slices = np.round(np.linspace(0,I.shape[1],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1)

    # get slices
    RGB_ = RGB[:,slices[i]].transpose(1,2,0)
    S_ = S[0,slices[i]]
    Jt_ = np.copy(Jt[0,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])

slices = np.round(np.linspace(0,I.shape[2],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1 + n)

    # get slices
    RGB_ = RGB[:,:,slices[i]].transpose(1,2,0)
    S_ = S[0,:,slices[i]]
    Jt_ = np.copy(Jt[0,:,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])

slices = np.round(np.linspace(0,I.shape[3],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1 + n+n)

    # get slices
    RGB_ = RGB[:,:,:,slices[i]].transpose(1,2,0)
    S_ = S[0,:,:,slices[i]]
    Jt_ = np.copy(Jt[0,:,:,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle('Atlas space')
fig.subplots_adjust(wspace=0,hspace=0,left=0.0,right=1,bottom=0,top=0.95)
fig.savefig(os.path.join(output_prefix, 'atlas_space.jpg'), **figopts)

# view the transformed labels with the target
fig = plt.figure(figsize=(8,5))
slices = np.round(np.linspace(0,J.shape[1],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1)

    # get slices
    RGB_ = RGBt[:,slices[i]].transpose(1,2,0)
    S_ = St[0,slices[i]]
    Jt_ = np.copy(J[0,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])

slices = np.round(np.linspace(0,J.shape[2],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1 + n)

    # get slices
    RGB_ = RGBt[:,:,slices[i]].transpose(1,2,0)
    S_ = St[0,:,slices[i]]
    Jt_ = np.copy(J[0,:,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])

slices = np.round(np.linspace(0,J.shape[3],n+2)[1:-1]).astype(int)
for i in range(n):
    ax = fig.add_subplot(3,n,i+1 + n+n)

    # get slices
    RGB_ = RGBt[:,:,:,slices[i]].transpose(1,2,0)
    S_ = St[0,:,:,slices[i]]
    Jt_ = np.copy(J[0,:,:,slices[i]])
    #Jt_ -= np.quantile(Jt_,0.01)
    Jt_ -= minval
    #Jt_ /= np.quantile(Jt_,0.99)
    Jt_ /= maxval-minval
    # find boundaries
    border = S_ != np.roll(S_,shift=1,axis=0)
    border |= S_ != np.roll(S_,shift=-1,axis=0)
    border |= S_ != np.roll(S_,shift=1,axis=1)
    border |= S_ != np.roll(S_,shift=-1,axis=1)

    # draw
    ax.imshow(alpha*border[...,None]*RGB_ + ((1-alpha*border)*Jt_)[...,None])
    if i>0:
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle('Target space')
fig.subplots_adjust(wspace=0,hspace=0,left=0.0,right=1,bottom=0,top=0.95)
fig.savefig(os.path.join(output_prefix, 'target_space.jpg'), **figopts)


### Get bounding boxes for striatum or another structure

# bounding boxes using St
# ontology_name = os.path.join(output_prefix, 'allen_ontology.csv')
# r = 'http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,\
# rma::criteria,[ontology_id$eq1],\
# rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]\
# '
# if True:# not os.path.exists(ontology_name):
#     output = requests.get(r)
#     with open(ontology_name,'wt') as f:
#         f.write(output.text)
        
ontology_name = '/panfs/dong/atlas_3d/upenn_vtk/atlas_info_KimRef_FPbasedLabel_v2.7.csv'

parent_column = 7  # 8 for allen, 7 for yongsoo
label_column = 0  # 0 for both
shortname_column = 2  # 3 for allen, 2 for yongsoo
longname_column = 1  # 2 for allen, 1 for yongsoo

ontology = dict()
with open(ontology_name) as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='"')
    count = 0
    for row in csvreader:
        if count == 0:
            headers = row
            print(headers)
        else:
            if not row[parent_column]:
                parent = -1
            else:
                parent = int(row[parent_column])
            ontology[int(row[label_column])] = (row[shortname_column],row[longname_column],parent)
        count += 1

# we need to find all the descendants of a given label
# first we'll get children
children = dict()
for o in ontology:
    parent = ontology[o][-1]
    if parent not in children:
        children[parent] = []
    children[parent].append(o)


# now we go from children to descendents
descendents = dict(children)
for o in descendents:
    for child in descendents[o]:
        if child in descendents: # if I don't do this i get a key error 0
            descendents[o].extend(descendents[child])
descendents[0] = []


descendents_and_self = dict(descendents)
for o in ontology:
    if o not in descendents_and_self:
        descendents_and_self[o] = [o]
    else:
        descendents_and_self[o].append(o)

# or we could just loop through
labels = np.unique(St)
bbox = dict()
for l in labels:
    # skip background
    if l == 0:
        continue

    Sl = St == l
    bbox2 = xJ[2][np.nonzero(np.sum(Sl,(0,1,2))>0)[0][[0,-1]]]
    bbox1 = xJ[1][np.nonzero(np.sum(Sl,(0,1,3))>0)[0][[0,-1]]]
    bbox0 = xJ[0][np.nonzero(np.sum(Sl,(0,2,3))>0)[0][[0,-1]]]
    bbox[l] = (bbox2[0],bbox2[1],bbox1[0],bbox1[1],bbox0[0],bbox0[1],ontology[l][0],ontology[l][1])

df = pd.DataFrame(bbox).T
bbox_headings = ('x0','x1','y0','y1','z0','z1','short name','long name')
df.columns=bbox_headings
df.index.name = 'id'
df.to_csv(os.path.join(output_prefix, 'bboxes.csv'))

# TODO, separate left from right in bounding box

# TODO, use a surface instead of bounding box. use marching cubes. figure out file format.
# see link on slack vrml (wireframe objects9-)

def compute_face_normals(verts,faces,normalize=False):
    e1 = verts[faces[:,1]] - verts[faces[:,0]]
    e2 = verts[faces[:,2]] - verts[faces[:,1]]
    n = np.cross(e1,e2)/2.0
    if normalize:
        n /= np.sqrt(np.sum(n**2,1,keepdims=True))
        pass
    return n

# let's get marching cubes surfaces going
oJ = [x[0] for x in xJ]
# or we could just loop through
labels = np.unique(St)
bbox = dict()

#verts0,faces0,normals0,values0 = marching_cubes((St[0]>0)*1.0,level=0.5,spacing=dJ)


fig = plt.figure()

# Generate descendent meshes
for l in ontology:
    print(f'starting {l}')
    # skip background
    if l == 0:
        print('skipping 0')
        continue

    Sl = St == l
    count0 = np.sum(Sl)
    # do marching cubes
    print('adding ',end='')
    for o in descendents_and_self[l]:
        print(f'{o},',end='')
        Sl = np.logical_or(Sl,St==o)
    count1 = np.sum(Sl)
    if count0 != count1:
        print(f'Structure {l} shows differences')
    if count1 == 0:
        print(f'no voxels for structure {l}')
        continue

    verts,faces,normals,values = marching_cubes(Sl[0]*1.0,level=0.5,spacing=dJ)
    # deal with the offsets
    verts += oJ
    
    # let's save this
    readme_dct = {'notes' : 'Data are saved in ZYX order',
                  'atlas_id' : ontology[l][1],
                  'id' : ontology[l][0] }
    readme = str(readme_dct)
    # Clean id to prevent region names interfering with file name
    clean_id = readme_dct["id"]
    for char in ['/', "' ", ', ', " "]:
        clean_id = clean_id.replace(char, '_')

    struct_des_fname = os.path.join(output_prefix, f'structure_AND_DESCENDENTS_{l:012d}_surface_{clean_id}.npz')
    np.savez(struct_des_fname, verts=verts,faces=faces,normals=normals,values=values,readme=readme, origin=origin)

    if jpg:
        surf = Poly3DCollection(verts[faces])
        n = compute_face_normals(verts,faces,normalize=True)
        surf.set_color(n*0.5+0.5)
        fig.clf()
        ax = fig.add_subplot(projection='3d')
        ax.add_collection3d(surf)
        xlim = (np.min(verts[:,0]),np.max(verts[:,0]))
        ylim = (np.min(verts[:,1]),np.max(verts[:,1]))
        zlim = (np.min(verts[:,2]),np.max(verts[:,2]))
        # fix aspect ratio
        r = [np.diff(x) for x in (xlim,ylim,zlim)]
        rmax = np.max(r)
        c = [np.mean(x) for x in (xlim,ylim,zlim)]
        xlim = (c[0]-rmax/2,c[0]+rmax/2)
        ylim = (c[1]-rmax/2,c[1]+rmax/2)
        zlim = (c[2]-rmax/2,c[2]+rmax/2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.set_title(f'structure {l}, {ontology[l][1]} ({ontology[l][0]})')
        fig.canvas.draw()
        fig.savefig(os.path.join(output_prefix, f'structure_AND_DESCENDENTS_{l:012d}_surface.jpg'))
