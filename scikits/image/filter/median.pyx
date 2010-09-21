import numpy as np
cimport numpy as np
import scikits.image.utils.shapes as shapes

cdef extern from "c_src/c_median.c":
    int c_median_iteration(unsigned char*, int, int, unsigned char*, int*, int)

def median(im, selem=None, r=1.0, n=1):
    """
    Perform a median on a uint8 array

    Parameters
    ----------
    im: ndarray
        input data to be denoised

    selem: structuring element, optional
        structuring element to use when denoising, or None for a
        disc of radius r

    r: float, optional
        if selem is None, this is the radius of the disc element 
        used when denoising

    n: int, optional
        number of filtering iterations
    """
    if r <= 0.0:
        raise ValueError('r should be > 0')
    if n <= 0:
        raise ValueError('n should be >= 1')
    if len(im.shape) != 2:
        raise ValueError('Expecting a 2d image')

    if shape is None:
        shape = shapes.disc(r)

    cdef int nx = im.shape[0]
    cdef int ny = im.shape[1]

    cdef np.ndarray[np.uint8_t] flat_im = im.ravel()
    cdef np.ndarray[np.int32_t] shapec = shapes.shape_coords(shape).ravel()

    cdef np.ndarray[np.uint8_t, mode='c'] contim
    try:
        contim = flat_im
    except:
        contim = flat_im.astype(np.uint8)

    cdef np.ndarray[np.int32_t, mode='c'] contshapec
    try:
        contshapec = shapec
    except:
        contshapec = shapec.astype(np.int32)

    cdef np.ndarray[np.uint8_t] v = np.zeros_like(contim)
    cdef np.ndarray[np.uint8_t] w
    if n != 1:
        w = np.zeros_like(contim)
    else:
        w = None
    # Number of pixels changed in the current iteration
    cdef int n_changed = 1
    cdef np.ndarray[np.uint8_t] src = contim
    i = 0
    while i < n:
        if src is not contim:
            src[:] = v
            n_changed = c_median_iteration(<unsigned char *>src.data, \
                nx, ny, <unsigned char *>v.data, <int *>contshapec.data, \
                                    len(shapec)/2)
    #    print "iteration {0}: {1} grey level(s) modified".format(i+1, n_changed)
        src = w
        if n_changed == 0:
            break
        i += 1

    #  if n_changed == 0:
    #    print "median blocked after {0} effective iteration(s)".format(i)

    return v.reshape((nx, ny))
