import numpy as np
cimport numpy as np
import scikits.image.utils.shapes as shapes

cdef extern from "c_src/c_median.c":
    int c_median_iteration(unsigned char*, int, int, unsigned char*, int*, int)

def median(im, selem=None, radius=1.0, niter=1):
    """
    Perform a median on a uint8 array

    Parameters
    ----------
    im : ndarray
        input data to be denoised

    selem : structuring element, optional
        structuring element to use when denoising, or None for a disc 
        of radius r

    radius : float, optional
        if selem is None, this is the radius of the disc element used 
        when denoising

    niter : int, optional
        number of filtering iterations

    Returns
    -------
    v : ndarray
        The denoised image
    
    Notes
    -----
    The structuring element is a two-dimensional array containing non-null 
    values on the points included in the element (the center is assumed to 
    be the point at (width/2, height/2)).

    Examples
    --------
    >>> import scipy
    >>> lena = scipy.lena()
    >>> lena_denoised = median(lena.astype(np.uint8), niter=5)
    """
    if radius <= 0.0:
        raise ValueError('radius should be > 0')
    if niter <= 0:
        raise ValueError('niter should be >= 1')
    if len(im.shape) != 2:
        raise ValueError('Expecting a 2d image')

    if shape is None:
        shape = shapes.disc(radius)

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
    if niter != 1:
        w = np.zeros_like(contim)
    else:
        w = None

    # Number of pixels changed in the current iteration
    cdef int n_changed = 1
    cdef np.ndarray[np.uint8_t] src = contim
    for i in xrange(niter):
        if src is not contim:
            src[:] = v
        n_changed = c_median_iteration(<unsigned char *> src.data, nx, ny,
            <unsigned char *> v.data, <int *>contshapec.data, len(shapec) / 2)
        src = w
        if n_changed == 0:
            break

    return v.reshape((nx, ny))
