# encoding: utf-8
import numpy as np
cimport numpy as np
cimport cython
from numpy.fft import fft2, ifft2

# Utility function that converts an image to float32 when required
def _ensures_float32(im):
    if not np.issubdtype(im.dtype, np.float32):
        return im.astype(np.float32)
    else:
        return im

@cython.boundscheck(False)
@cython.wraparound(False)
def perdecomp(u):
    if len(u.shape) != 2:
        raise ValueError('Expecting a 2d array')

    cdef np.ndarray[np.float32_t, ndim=2] fu = _ensures_float32(u)

    cdef int nx, ny
    ny, nx = fu.shape[0], fu.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] s = np.zeros_like(fu)

    cdef int x, y
    cdef float b
    for x in xrange(nx):
        b = fu[0, x] - fu[ny-1, x]
        s[0, x] += b
        s[ny-1, x] -= b

    for y in xrange(ny):
        b = fu[y, 0] - fu[y, nx-1]
        s[y, 0] += b
        s[y, nx-1] -= b

    cdef np.ndarray[np.complex_t, ndim=2] t = fft2(s)
    t[0,0].real = 0.0
    t[0,0].imag = 0.0

    cdef float cx = 2.0*np.pi/float(nx)
    cdef float cy = 2.0*np.pi/float(ny)

    cdef float w
    cdef np.ndarray[np.float32_t, ndim=1] v2 = np.zeros(ny-1, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] v3

    cos = np.cos

    for y in xrange(1,ny):
        v2[y-1] = ny-y if y>ny/2 else y
    v2 = 1.0-cos(cy*v2)

    for x in xrange(nx):
        w = nx-x if x>nx/2 else x
        w = 1.0-cos(cx*w)
        v3 = v2 + w
        v3 = 0.5/v3
        t[1:,x] *= v3

    s = np.real(ifft2(t)).astype(np.float32)
    p = u-s

    return (p, s)
