import numpy as np
import matplotlib.pyplot as plt
import time

def draw_slices(I,xI=None,n=5,channel_ax=0,fig=None,vmin=None,vmax=None,**kwargs):
    ''' Draw slices in 3 orientations

    Parameters
    ----------
    I : array
        Array of imaging data. Assumed to be multi channel of the form C x row x col lx slice

    xI :

    channel_ax : int or None
        Which axis stores the channels, if any. Defaults to 0, which is the pytorch convention



    Returns
    -------
    fig : matplotlib figure
        Figure object drawn into

    ax : array of matplotlib axes
        Axes as a 3xn array.

    '''

    # Identify channel axes and move it to end
    if channel_ax is None:
        I = I[...,None]
    else:
        # swap it with a dummy axis at the end, then squeeze out the dummy axis
        I = np.swapaxes(I[...,None],channel_ax,-1).squeeze()
        # check the number of channels and make sure it is 3
        nc = I.shape[-1]
    
    # check normalization to do it consistently
    if vmin is None:
        vmin = np.quantile(I,0.001)
    if vmax is None:
        vmax = np.quantile(I,0.999)
    I -= vmin
    I /= (vmax-vmin)



    # set up a figure
    if fig is None:
        fig = plt.figure()
    
    # now we'll set up slices along each axis
    ax = []
    inds = np.round(np.linspace(0,I.shape[0],n+2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3,n,i+1)
        ax__.imshow(I[inds[i]],vmin=0,vmax=1)
        ax_.append(ax__)
    ax.append(ax_)
    # second axis
    inds = np.round(np.linspace(0,I.shape[1],n+2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3,n,i+1+n)
        ax__.imshow(I[:,inds[i]],vmin=0,vmax=1)
        ax_.append(ax__)
    ax.append(ax_)
    # third axis
    inds = np.round(np.linspace(0,I.shape[2],n+2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3,n,i+1+n*2)
        ax__.imshow(I[:,:,inds[i]],vmin=0,vmax=1)
        ax_.append(ax__)
    ax.append(ax_)



    return fig,ax




def downsample_ax(I,d,ax):
    '''
    Downsample along a given axis
    
    Use np take so that ax can be an argument 
    and no fancy indexing is needed

    Note downsampling here leaves the end off.

    Parameters
    ----------
    I : numpy array
        Image to be downsampled along one axis
    d : int
        Downsampling factor
    ax : int
        Axis to be downsampled on


    Returns
    -------
    Id : numpy array
        Image downsampled by averaging on given axis.
        
    '''
    if d == 1:
        return np.array(I) # note this is making a new array, just like below
    
    nId = np.array(I.shape)
    nId[ax]//=d
    Id = np.zeros(nId,dtype=I.dtype)
    for i in range(d):
        Id += np.take(I,np.arange(i,nId[ax]*d,d),ax)/d        
    return Id

def is_prime(d):
    '''
    Determine if an integer is prime by looping over all its possible factors.
    This is not a high performance algorithm and should be done with small numbers
    
    Parameters
    ----------
    d : int
        An integer to test if it is prime

    Returns
    -------
    is_prime : bool
        Whether or not the number is prime
    '''
    for i in range(2,d): # between 2 and d-2
        if d//i == d/i:
            return False
    return True
def prime_factor(d):
    '''
    Input an integer and return a list of prime factors.

    Parameters
    ----------
    d : int
        An integer to factorize

    Returns
    -------
    factors : list of int
        A list of prime factors sorted from lowest to highest

    '''
    if is_prime(d):
        return [d]
    # otherwise I'll recurse and get a left factor and a right factor
    for i in range(2,d):
        if d//i == d/i:
            # this is a factor
            left = i
            right = d//i
            output = []
            output += prime_factor(left)
            output += prime_factor(right)
    output.sort()
    return output
    
         
    
def downsample(I,d):
    '''
    Downsample along each axis
    
    If it can be factored it would be really good
    
    In factors always put the small number first

    Not pixels are left off the end.

    Parameters
    ----------
    I : numpy array
        Imaging data to downsample
    d : list of ints
        Downsampling factors along each axis



    Returns
    -------
    Id : numpy array
        Downsampled image

    '''
    Id = I
    for i,di in enumerate(d):        
        for dii in prime_factor(di):            
            Id = downsample_ax(Id,dii,i)        
    return Id


def downsample_dataset(data,down,xI=None):
    # okay now I have to iterate over the dataset
    fig,ax = plt.subplots(1,2)
    working = []
    output = []
    start = time.time()
    for i in range(data.shape[0]):
        starti = time.time()
        
        s = data[i]
        sd = downsample(s.astype(float),down[1:])
        ax[0].cla()
        ax[0].imshow(sd)
        working.append(sd)
        if len(working) == down[0]:
            out = downsample(np.stack(working),[down[0],1,1])        
            ax[1].cla()
            ax[1].imshow(out[0])
            output.append(out)
            working = []
        fig.canvas.draw()
        print(f'Finished loading slice {i} of {data.shape[0]}, time {time.time() - starti} s')
    output = np.concatenate(output)        
    Id = output

    output = Id
    if xI is not None:
        xId = [downsample(x,[d]) for x,d in zip(xI,down)]
        output = (Id,xId)
    

    print(f'Finished downsampling, time {time.time() - start}')

    return output

def imaris_bytes_to_float(data):
    return float(''.join([c.decode() for c in data]))

def imaris_get_pixel_size(f):
    '''
    Get pixel size from Imaris dataset using attributes
    '''
    info = f['DataSetInfo/Image']
    dx0 = (imaris_bytes_to_float(info.attrs['ExtMax0']) - imaris_bytes_to_float(info.attrs['ExtMin0']))/(imaris_bytes_to_float(info.attrs['X']))
    dx1 = (imaris_bytes_to_float(info.attrs['ExtMax1']) - imaris_bytes_to_float(info.attrs['ExtMin1']))/(imaris_bytes_to_float(info.attrs['Y']))
    dx2 = (imaris_bytes_to_float(info.attrs['ExtMax2']) - imaris_bytes_to_float(info.attrs['ExtMin2']))/(imaris_bytes_to_float(info.attrs['Z']))
    return np.array([dx2,dx1,dx0]) # note here we need to swap the order to be consistent with zyx images
def imaris_get_x(f):
    '''
    Get 3D pixel locations
    
    Note when we use X Y Z metadata, this may be smaller than the 
    number of voxels (due to padding for chunks). This does correspond
    to the real acquired data though.
    
    
    
    Parameters
    ----------
    f : imaris file
        Imaris file to read info from
        
    Returns
    -------
    x : list of numpy arrays
        locations of each pixel along z,y,x directions
    '''
    info = f['DataSetInfo/Image']
    xmin = np.array(
        [imaris_bytes_to_float(info.attrs['ExtMin2']),
         imaris_bytes_to_float(info.attrs['ExtMin1']),
         imaris_bytes_to_float(info.attrs['ExtMin0'])]
    )
    xmax = np.array(
        [imaris_bytes_to_float(info.attrs['ExtMax2']),
         imaris_bytes_to_float(info.attrs['ExtMax1']),
         imaris_bytes_to_float(info.attrs['ExtMax0'])]
    )
    nx = np.array([imaris_bytes_to_float(info.attrs['Z']),
                   imaris_bytes_to_float(info.attrs['Y']),
                   imaris_bytes_to_float(info.attrs['X'])])
    dx = (xmax - xmin)/nx
    x = [np.arange(n)*d + x0 for n,d,x0 in zip(nx,dx,xmin)]
    return x
    