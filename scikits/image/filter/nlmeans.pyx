import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_src/c_nlmeans.c":
    int c_nlmeans(float *, float *, int, int, float, int, float, int, float, unsigned char *, int, int)

def nlmeans(arr, float h=10.0, int s=7, a=None, int d=10, float c=1.0, mask=None):
    """
    Perform the non-local means filter

    Parameters
    ----------
    arr: ndarray
        input data to be denoised

    h: float, optional
       regularization parameter

    s: int, optional
       patch side length (an odd integer)

    a: float, optional
       decay of Euclidean patch distance, default (s-1)/4

    d: int, optional
       maximum patch distance

    c: float, optional
       weight for self-patch

    mask: ndarray, optional
          do not denoise pixels x with mask(x) == 0

    Returns
    -------
    out : ndarray
        The filtered image

    References
    ----------

    Examples
    --------
    >>> import scipy
    >>> lena = scipy.lena()
    >>> lena_filtered = nlmeans(lena)
    """
    if len(arr.shape) != 2:
        raise ValueError('Expecting a 2d array')
    cdef int nx = arr.shape[0]
    cdef int ny = arr.shape[1]

    if s%2 != 1:
        raise ValueError('s must be an odd integer')

    if a is None:
        a = float(s-1)/4.0

    cdef int mx=0, my=0
    cdef unsigned char *pmask = cython.NULL
    cdef np.ndarray[np.uint8_t, mode='c'] fmask
    if mask is not None:
        if len(mask.shape) != 2:
            raise ValueError('mask should be a 2d-array')
        else:
            mx, my = mask.shape[0], mask.shape[1]
            flat_mask = mask.ravel()
            fmask = flat_mask.astype(np.uint8)
            pmask = <unsigned char *>fmask.data

    flat_arr = arr.ravel()
    cdef np.ndarray[np.float32_t, mode='c'] contarr = flat_arr.astype(np.float32)
    cdef np.ndarray[np.float32_t] out = np.zeros_like(contarr)

    err_code = c_nlmeans(<float *>contarr.data, <float *>out.data, nx, ny,
            h, s, <float>a, d, c, <unsigned char *>pmask, mx, my)
    if err_code == -1:
        raise ValueError('Parameter error')
    elif err_code == 1:
        raise Exception('Not enough memory')
    elif err_code != 0:
        raise Exception('c_nlmeans exited with code ' + str(err_code))

    return out.reshape((nx, ny))
