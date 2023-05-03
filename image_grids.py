#
#
#
#
#

import numpy as np
import cv2

import matplotlib.pyplot as plt

#--- For scatter_images()
from skimage.transform import resize
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#========================================================
#  Simple routine to show a pair of images
#========================================================
def show_image_pair(_ima1, _ima2):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(_ima1, cmap='gray')
    ax[1].imshow(_ima2, cmap='gray')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

#========================================================
#  Simple routine to show a row of n images
#
#  _imas is a list of images
#========================================================
def show_image_row(_imas, plot_size=None, interpolation='nearest'):

    if (plot_size != None): plt.rcParams['figure.figsize'] = [plot_size[0], plot_size[1]]

    ncols = len(_imas)
    fig, axes = plt.subplots(nrows=1, ncols=ncols)
    ax = axes.ravel()

    for i in range(ncols):
        ax[i].imshow(_imas[i], cmap='gray',interpolation=interpolation)
        ax[i].axis('off')
        
    plt.tight_layout()
    plt.show()


#========================================================
#   Create scatterplot with images instead of points directly
#      into an image. The input images must be normalized in
#      the range [0-1]
#
#   Inputs:
#     _x,_y  : position of tiles, normalized to 0-1
#     _tiles : list of images (as numpy arrays with range 0-1)
#     _tile_size: maximum tile size
#     _width,_height: size of output image
#
#========================================================
def scatter_images_pil(_x, _y, _tiles, _tile_size=100,_width=800,_height=800):
    
    full_image = Image.new('RGB', (_width, _height))
    for idx, x in enumerate(_tiles):
        tile = Image.fromarray(np.uint8(x * 255))
        rs = max(1, tile.width / _tile_size, tile.height / _tile_size)
        tile = tile.resize((int(tile.width  / rs),
                            int(tile.height / rs)),
                           Image.ANTIALIAS)
        full_image.paste(tile, (int((_width  - _tile_size) * _x[idx]),
                                int((_height - _tile_size) * _y[idx])))
    return full_image
    

#========================================================
#  Helper function for plotting images at a given position inside
#  a matplotlib plot
#========================================================
def im_scatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    #try:
    #    image = plt.imread(image)
    #except TypeError:
    #    # Likely already an array...
    #    pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    return artists


#========================================================
#  Show images as points in a scatter plot. 
#  Input:
#    _x, _y: list of images positions
#    _image_array: list of images (same number of elements as _x,_y)
#  Notes:
#    Use function load_images_folder() for _image_array
# 
#========================================================
def scatter_images(_x, _y, _image_array, xlim=(0,10), ylim=(0,10), plot_size=(10,10),  zoom=1, background='white'):

    plt.rcParams['figure.figsize'] = [plot_size[0], plot_size[1]]
    plt.rcParams['figure.facecolor'] = background
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_facecolor(background)

    for i, image_i in enumerate(_image_array):       
        im_scatter(_x[i], _y[i], image_i, zoom=zoom, ax=ax)
        
    plt.show()


#========================================================
#
#  Make a grid of images
#
#========================================================
def make_image_grid(imarray, nx, ny, margin=2, labels=None, start=0, rebin=None,background=0):

    im_shape = imarray.shape
    n_im = im_shape[0]
    im_w = im_shape[1]
    im_h = im_shape[2]
    
    width  = nx * im_w + (nx-1) * margin
    height = ny * im_h + (ny-1) * margin

    if len(im_shape) == 4:
        stitched_filters = np.zeros((width, height,3)) + background
    else: 
        stitched_filters = np.zeros((width, height)) + background
        
    #--- Sanity check and exit if neccesary
    if labels is not None:
        if n_im != len(labels):
            print("Images and labels have different number of elements!!!, exiting empty image...")
            return stitched_filters

    #--- Start even if not used
    font = cv2.FONT_HERSHEY_SIMPLEX
            
    #--- Fill grid
    cont=0
    for j in range(ny):
        for i in range(nx):

            ind_ij = cont+start
            
            #--- In case there are less images than grid elements
            if (ind_ij >= n_im): continue

            #--- Got obscure error and found I had to copy the array
            #    https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
            #    I suspect it has to do with slicing not copying the data to the sliced array
            image_ij = np.array(imarray[ind_ij,:])#.transpose([1,2,0]))
            image_ij = image_ij.copy()
            
            if labels is not None:
                text = str(labels[ind_ij])
                cv2.putText(image_ij, text, (0,im_h), font, 1, (255, 255, 255))

            #--- Add this image to grid
            if len(im_shape) == 4:
                stitched_filters[(im_w + margin) * i: (im_w + margin) * i + im_w,
                                 (im_h + margin) * j: (im_h + margin) * j + im_h, :] = image_ij
            else:
                stitched_filters[(im_w + margin) * i: (im_w + margin) * i + im_w,
                                 (im_h + margin) * j: (im_h + margin) * j + im_h] = image_ij
	
            cont = cont+1
            
    return stitched_filters
        
