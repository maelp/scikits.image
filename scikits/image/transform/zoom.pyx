# encoding: utf-8

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_src/c_zoom.c":
    int _c_zoom(float*, int, int, float, float, float*, int, int, float, float, int)

class Zoom(object):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = -3
    SPLINE_3 = 3
    SPLINE_5 = 5
    SPLINE_7 = 7
    SPLINE_9 = 9
    SPLINE_11 = 11

def _ensures_float(im):
    if not np.issubdtype(im.dtype, float):
        return im.astype(np.float32)
    else:
        return im

def zoom(im, order, float x, float y, int w, int h, float scale, float bgcolor):
    """
    Perform a zoom and image cropping

    Parameters
    ----------
    im: ndarray (2d or 3d array of any type)
        input data

    order: int
        zooming method
        0: constant (nearest-neighbor)
        1: bilinear
        -3: bicubic
def _ensures_float(im):
    if not np.issubdtype(im.dtype, float):
        return im.astype(np.float32)
    else:
        return im
        3,5,7,9,11: spline of order 3,5,7,9,11

    x, y: float
        coordinates (in the input image) of the sample corresponding to the
        first pixel of the output

    w, h: int
        desired size of the output image

    scale: float
        zoom value (> 0, scale = 1.0: no zoom)

    bgcolor: float
        value of the background (when interpolating pixels near the image border)

    Returns
    -------
    out : ndarray (float32 2d-array)
        The zoomed and cropped image

    Examples
    --------
    >>> import scipy
    >>> lena = scipy.lena()
    >>> # Zoom bilinear, 2x, with a sub-pixel translation, and output a 100x100 image
    >>> lena_zoomed = fzoom(lena, Zoom.BILINEAR, 0.5, 0.5, 100, 100, 2.0, 0.0)
    """
    if im.ndim == 2:
        panes = [_ensures_float(im)]
    elif im.ndim == 3:
        if im.shape[2] == 1:
            panes = [_ensures_float(im[:,:,0])]
        elif im.shape[2] == 3 or im.shape[2] == 4:
            imR = _ensures_float(im[:,:,0])
            imG = _ensures_float(im[:,:,1])
            imB = _ensures_float(im[:,:,2])
            panes = [imR, imG, imB]
        else:
            raise ValueError('Invalid number of color panes')
    else:
        raise ValueError('The im must be either a 2D or 3D array')

    for i, pane in enumerate(panes):
        panes[i] = fzoom(pane, order, x, y, w, h, scale, bgcolor)

    if len(panes) == 1:
        return panes[0]
    else:
        out = np.empty((h,w,3), dtype=np.float32)
        out[:,:,0] = panes[0]
        out[:,:,1] = panes[1]
        out[:,:,2] = panes[2]
        return out

def fzoom(np.ndarray[np.float32_t, ndim=2]im, int order,
        float x, float y, int w, int h, float scale, float bgcolor):
    """
    Perform a zoom and image cropping on a floating-point single-pane image

    Parameters
    ----------
    im: ndarray (float32 2d-array)
        input data

    order: int
        zooming method
        0: constant (nearest-neighbor)
        1: bilinear
        -3: bicubic
        3,5,7,9,11: spline of order 3,5,7,9,11

    x, y: float
        coordinates (in the input image) of the sample corresponding to the
        first pixel of the output

    w, h: int
        desired size of the output image

    scale: float
        zoom value (> 0, scale = 1.0: no zoom)

    bgcolor: float
        value of the background (when interpolating pixels near the image border)

    Returns
    -------
    out : ndarray (float32 2d-array)
        The zoomed and cropped image

    Examples
    --------
    >>> import scipy
    >>> lena = scipy.lena().astype(np.float32)
    >>> # Zoom bilinear, 2x, with a sub-pixel translation, and output a 100x100 image
    >>> lena_zoomed = fzoom(lena, Zoom.BILINEAR, 0.5, 0.5, 100, 100, 2.0, 0.0)
    """

    if w <= 0 or h <= 0:
        raise ValueError('w and h should be >= 1')
    if scale <= 0.0:
        raise ValueError('scale should be > 0')
    if order not in [-3,0,1,3,5,7,9,11]:
        raise ValueError('unknown zoom order')

    cdef int ny = im.shape[0]
    cdef int nx = im.shape[1]

    cdef np.ndarray[np.float32_t] flat_im = im.ravel()

    cdef np.ndarray[np.float32_t, mode='c'] contim
    try:
        contim = flat_im
    except:
        contim = flat_im.astype(np.float32)

    cdef np.ndarray[np.float32_t] out = np.empty(h*w, dtype=np.float32)

    c = _c_zoom(<float*>contim.data, nx, ny, x, y,
            <float*>out.data, w, h, scale, bgcolor, order)
    if c != 0:
        if c == 1:
            raise Exception("Not enough memory")
        elif c == -1:
            raise ValueError("Invalid parameters")
        else:
            raise Exception("_c_zoom returned error code "+str(c))

    return out.reshape((h, w))
