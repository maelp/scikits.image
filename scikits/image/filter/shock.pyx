import numpy as np
cimport numpy as np

cdef extern from "c_src/c_shock.c":
    int c_shock(float *, float *, int, int, int, float)

def shock(arr, int niter=10, float s=0.1):
    """
    Perform the Rudin shock filter

    Parameters
    ----------
    arr: ndarray
        input data to be denoised

    niter: int, optional
        number of iterations

    s: float in [0.0, 1.0], optional
        scale step

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
    >>> lena_filtered = shock(lena, niter=5)
    """
    if len(arr.shape) != 2:
        raise ValueError('Expecting a 2d array')
    cdef int nx = arr.shape[0]
    cdef int ny = arr.shape[1]

    s = min(1.0, max(0.0, s))

    flat_arr = arr.ravel()
    cdef np.ndarray[np.float32_t, mode='c'] contarr = flat_arr.astype(np.float32)
    cdef np.ndarray[np.float32_t] out = np.zeros_like(contarr)

    err_code = c_shock(<float *> contarr.data, <float *>out.data, nx, ny,
                            niter, s)
    if err_code != 0:
        raise RuntimeError('c_shock exited with code ' + str(err_code))

    return out.reshape((nx, ny))
