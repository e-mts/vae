#
#
#

import numpy as np
import matplotlib.pyplot as plt
import cv2

#--- For scatter_images()
from skimage.transform import resize
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import scipy.interpolate
import scipy.ndimage

from bisect import bisect_left


#========================================================
#  drAll_id contains ids froma  large sample of galaxies
#  dr7_id is the sample we are going to find in drAll_id
#
#  #--- Instantiate an object from the class
#  searched = search.SearchBisect(drAll_id)
#  #--- Perform a search for element i in dr7_id
#  found = searched.find_item(dr7_id[i])
#
#  History:
#    - 13/05/2021 Fixed bug in find_item() to return -1 on failed search
#
#========================================================
class SearchBisect:

    def __init__(self, _data):
        self.data = np.copy(_data)
        self.data_sort_ind = np.argsort(self.data)
        self.data_sorted   = np.sort(self.data)
        
    def search_bisect(alist, item):
        i = bisect_left(alist, item)
        if i != len(alist) and alist[i] == item:
            return i
        else:
            return -1
                    
    def find_item(self, _item):
        indx  = search_bisect(self.data_sorted, _item)
        if indx == -1:
            return -1
        else:
            return self.data_sort_ind[indx]



#========================================================
#
#========================================================
def subsample_dict(cat, sub):

    newcat = {}
    for key in cat.keys():
        newcat[key] = cat[key][sub]
    
    return newcat
        
#========================================================
#   Simple replacement for where function, pass multiple conditions
#   as a list
#========================================================
def where(_arr_list):

    _all_list = list(_arr_list)
    for i in range(len(_arr_list)):
        if (i==0):
            result = _arr_list[i]
        else:
            result = result * _arr_list[i]
                   
    return result.nonzero()[0]
    


#========================================================
#
#========================================================
def info(_arr):
    type_arr = type(_arr)
    if type(_arr) is list:
        print('>>> type:', type(_arr))
        print('    len:', len(_arr))
        print('    min :', np.min(_arr))
        print('    max :', np.max(_arr))
        print('    mean:', np.mean(_arr))
    elif type(_arr) is tuple:
        print('>>> type:', type(_arr))
        print('    len:', len(_arr))
        print('    min :', np.min(_arr))
        print('    max :', np.max(_arr))
        print('    mean:', np.mean(_arr))
    elif type(_arr) is np.ndarray:
        print('>>> type:', type(_arr))
        print('    len:', len(_arr))
        print('    shape:', _arr.shape)
        print('    min :', np.min(_arr))
        print('    max :', np.max(_arr))
        print('    mean:', np.mean(_arr))
    else:
        print('>>> type:', type(_arr))


#========================================================
#  IDL replacement for rebin, from https://gist.github.com/derricw/95eab740e1b08b78c03f
#
#  see also this code: https://scipython.com/blog/binning-a-2d-array-in-numpy/  
#
#    def rebin(arr, new_shape):
#        shape = (new_shape[0], arr.shape[0] // new_shape[0],
#                 new_shape[1], arr.shape[1] // new_shape[1])
#        return arr.reshape(shape).mean(-1).mean(1)
#========================================================
def rebin(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


#========================================================
#   Replacement for congrid
#========================================================
def congrid(a, newdims, method='neighbour', centre=False, minusone=False):

    m1  = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    newdims = np.asarray( newdims, dtype=int )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return np.asarray(newa)


#========================================================
#   Replacement for dist
#========================================================
def dist(n):
    n2 = int((n)/2)
    xoff = 0.0
    yoff = 0.0
    distance = np.sqrt(np.asarray([[(x+xoff)**2 + (y+yoff)**2 for x in range(-n2,n2)] for y in range(-n2,n2)], dtype=np.float32))
    return distance


#========================================================
#   Simple image plotting routine trying to replicate IDL
#   Coyote's tvscale()
#'none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
#'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'.
#========================================================
def tvscale(img, wsize=-1,scale=0, invert=False, plot_size=(6,6), interpolation='nearest'):

    #plt.rcParams['figure.figsize'] = plot_size
    
    #--- Select image scaling
    if scale==0:
        im = img
    if scale==1:
        im = (img-np.min(img)) / (np.max(img)-np.min(img)).astype(float)

    #--- No frame
    fig = plt.figure(figsize=plot_size, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
        
    if invert==True:
        imgplot = plt.imshow(img,interpolation=interpolation, origin='lower',cmap='gray')
    else:
        imgplot = plt.imshow(img,interpolation=interpolation,cmap='gray')
        
    plt.axis('off')
    plt.show()


#========================================================
#   Simple image plotting routine trying to replicate IDL
#   Coyote's tvscale()
#========================================================
def tvscale_cv(ima, title=None,wsize=-1,scale=0):

    if title is None:
        title = "Press a key to exit"
    
    #--- Select image scaling
    if scale==0:
        im = ima
    if scale==1:
        im = (ima-np.min(ima)) / (np.max(ima)-np.min(ima)).astype(float)

    #--- Select interpolation
    if wsize ==-1:
        cv2.imshow(title, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))        
    else:
        cv2.imshow(title, cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),wsize,interpolation=cv2.INTER_NEAREST)) #--- OJO: remember (row,column) for numpy
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#========================================================
# Limited implementation of IDL's histogram()
#   in order to get the reverse indices
#
#  inds = np.random.randint(0,10,20)
#  hist_val, sort_array_ind, hist_cumul = idl.histogram_index_int(inds)
#
#  #--- Print the group ID of bin 5:
#  ind = 5
#  inds[sort_array_ind[hist_cumul[ind]:hist_cumul[ind]+hist_val[ind]]]
#  print(inds)
#
#  #--- Loop over non-zero histogram bins
#  for i in range(len(hist_val)):
#      #--- Loop inside bin
#      for j in range(hist_val[i]):   
#          print(i,  inds[sort_array_ind[hist_cumul[i]+j]])
#
#========================================================
def histogram_index_int(array):
    array_shape = array.shape
    #--- Easier to work in 2D
    array_flat = np.ravel(array)
    sort_array_ind = np.argsort(array_flat)
    #--- Get cumulative sum
    hist        = np.array(np.bincount(array_flat))
    #--- Only non-zero histogram bins
    hist_val    = np.array(hist[np.array(np.nonzero(hist))].ravel())
    n_hist_val  = hist_val.size
    #--- Running cumulative, first element zero
    #hist_cumul = np.cumsum(hist_val)
    hist_cumul    = np.roll(np.cumsum(hist_val),1)
    hist_cumul[0] = 0
    #--- Back to original, OJO: check if this is efficient
    array = np.reshape(array,array_shape)

    return hist_val, sort_array_ind, hist_cumul


#========================================================
#
#========================================================
def plot(_x, _y, xtitle='', ytitle='', eps=None, xrange=None, yrange=None, xsize=4, ysize=4, charsize=12, pointsize=1, show=True):
    ''
    #
    #  Replacement for IDL's (scatter) plot function. This uses only one line, which
    #     for simple plots is more than enough
    #
    #  Including latex in text:
    #     '$r_e$ (predicted)'
    #
    #  See for discussion on fonts:
    #     https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    #
    #
    #  Getting available fonts:
    #     import matplotlib.font_manager
    #     for name in matplotlib.font_manager.fontManager.ttflist:
    #         print(name)
    #
    #  USAGE:
    #
    #     x = np.random.uniform(0,1,1000)
    #     y = np.random.uniform(0,1,1000)
    #     idl.plot_scatter(x,y, charsize=12,xtitle='$x_e$', xrange=(0,0.5), xsize=5,ysize=4)
    ''

    fig = plt.figure(figsize=(xsize, ysize), dpi=120)

    #--- Default data limits
    if xrange == None:
        xrange = (np.min(_x), np.max(_x))
    if yrange == None:
        yrange = (np.min(_y), np.max(_y))
        
    csfont = {'fontname':'Times New Roman'}
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.rcParams.update({'font.size': charsize})
  
    plt.xlim(xrange)
    plt.ylim(yrange)

    plt.plot(_x, _y, color='black')

    plt.xlabel(xtitle,**csfont)
    plt.ylabel(ytitle,**csfont)

    if show == True:
        plt.show()

    if eps != None:
        fig.savefig(eps, bbox_inches='tight')




#========================================================
#
#========================================================
def plot_scatter(_x, _y, xtitle='', ytitle='', eps=None, xrange=None, yrange=None, xsize=4, ysize=4, charsize=12, pointsize=1, show=True):
    ''
    #
    #  Replacement for IDL's (scatter) plot function. This uses only one line, which
    #     for simple plots is more than enough
    #
    #  Including latex in text:
    #     '$r_e$ (predicted)'
    #
    #  See for discussion on fonts:
    #     https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    #
    #
    #  Getting available fonts:
    #     import matplotlib.font_manager
    #     for name in matplotlib.font_manager.fontManager.ttflist:
    #         print(name)
    #
    #  USAGE:
    #
    #     x = np.random.uniform(0,1,1000)
    #     y = np.random.uniform(0,1,1000)
    #     idl.plot_scatter(x,y, charsize=12,xtitle='$x_e$', xrange=(0,0.5), xsize=5,ysize=4)
    ''

    fig = plt.figure(figsize=(xsize, ysize), dpi=120)

    #--- Default data limits
    if xrange == None:
        xrange = (np.min(_x), np.max(_x))
    if yrange == None:
        yrange = (np.min(_y), np.max(_y))
        
    csfont = {'fontname':'Times New Roman'}
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.rcParams.update({'font.size': charsize})
  
    plt.xlim(xrange)
    plt.ylim(yrange)

    plt.scatter(_x, _y, color='black', s=pointsize, marker='.')

    plt.xlabel(xtitle,**csfont)
    plt.ylabel(ytitle,**csfont)

    if show == True:
        plt.show()

    if eps != None:
        fig.savefig(eps, bbox_inches='tight')


#========================================================
#
#========================================================
def plot_scatter_fast(_x, _y, nx, ny, xtitle='', ytitle='', eps=None, xrange=None, yrange=None, xsize=4, ysize=4, charsize=12, pointsize=1):
    ''
    #
    #  Replacement for IDL's (scatter) plot function. This uses only one line, which
    #     for simple plots is more than enough
    #
    #  Including latex in text:
    #     '$r_e$ (predicted)'
    #
    #  See for discussion on fonts:
    #     https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    #
    #
    #  Getting available fonts:
    #     import matplotlib.font_manager
    #     for name in matplotlib.font_manager.fontManager.ttflist:
    #         print(name)
    #
    #  USAGE:
    #
    #     x = np.random.uniform(0,1,1000)
    #     y = np.random.uniform(0,1,1000)
    #     idl.plot_scatter(x,y, charsize=12,xtitle='$x_e$', xrange=(0,0.5), xsize=5,ysize=4)
    ''

    fig = plt.figure(figsize=(xsize, ysize), dpi=120)

    #--- Default data limits
    if xrange == None:
        xrange = (np.min(_x), np.max(_x))
    if yrange == None:
        yrange = (np.min(_y), np.max(_y))

        
        
    csfont = {'fontname':'Times New Roman'}
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.rcParams.update({'font.size': charsize})
  
    plt.xlim(xrange)
    plt.ylim(yrange)

    plt.scatter(_x, _y, color='black', s=pointsize, marker='.')

    plt.xlabel(xtitle,**csfont)
    plt.ylabel(ytitle,**csfont)

    plt.show()

    if eps != None:
        fig.savefig(eps, bbox_inches='tight')


#========================================================
#
#========================================================
def histo_3d(_x,_y,_z, _xrange, _yrange, _zrange, _ng, weights = None):

    #--- Only particles inside the box
    val = np.where( (_x >= _xrange[0]) * (_x < _xrange[1]) * (_y >= _yrange[0]) * (_y < _yrange[1]) * (_z >= _zrange[0]) * (_z < _zrange[1]) )[0]
    
    #--- Convert to coordinates
    xg = ( (_x[val] - _xrange[0]) / (_xrange[1] - _xrange[0]) * _ng ).astype(np.uint32)
    yg = ( (_y[val] - _yrange[0]) / (_yrange[1] - _yrange[0]) * _ng ).astype(np.uint32)
    zg = ( (_z[val] - _zrange[0]) / (_zrange[1] - _zrange[0]) * _ng ).astype(np.uint32)
    
    n = len(val)
    vol = np.zeros((_ng,_ng,_ng), dtype=np.float32)
    
    if weights is None:
        for i in range(n):
            vol[xg[i], yg[i], zg[i]] += 1
    else:
        for i in range(n):
            vol[xg[i], yg[i], zg[i]] += weights[val[i]]
        
    return vol

#========================================================
#
#========================================================
def histo_2d(_x, _y, _xrange, _yrange, _ng, weights = None):

    #--- Only particles inside the box
    val = np.where( (_x >= _xrange[0]) * (_x < _xrange[1]) * (_y >= _yrange[0]) * (_y < _yrange[1]) )[0]
    
    #--- Convert to coordinates
    xg = ( (_x[val] - _xrange[0]) / (_xrange[1] - _xrange[0]) * _ng ).astype(np.uint32)
    yg = ( (_y[val] - _yrange[0]) / (_yrange[1] - _yrange[0]) * _ng ).astype(np.uint32)
    
    n = len(val)
    vol = np.zeros((_ng,_ng), dtype=np.float32)
    
    if weights is None:
        for i in range(n):
            vol[xg[i], yg[i]] += 1
    else:
        for i in range(n):
            vol[xg[i], yg[i]] += weights[val[i]]
        
    return vol
