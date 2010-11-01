import numpy as np
cimport numpy as np
cimport cython

cdef extern from "c_src/c_nlmeans.c":
    int c_nlmeans(float *, float *, int, int, float, int, float, int, float)

def nlmeans(arr, float h=10.0, int s=7, a=None, int d=10, float c=1.0):
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

    flat_arr = arr.ravel()
    cdef np.ndarray[np.float32_t, mode='c'] contarr = flat_arr.astype(np.float32)
    cdef np.ndarray[np.float32_t] out = np.zeros_like(contarr)

    err_code = 0
    with nogil:
        err_code = c_nlmeans(<float *>contarr.data, <float *>out.data, nx, ny,
                h, s, <float>a, d, c)
    if err_code == -1:
        raise ValueError('Parameter error')
    elif err_code == 1:
        raise Exception('Not enough memory')
    elif err_code != 0:
        raise Exception('c_nlmeans exited with code ' + str(err_code))

    return out.reshape((nx, ny))
