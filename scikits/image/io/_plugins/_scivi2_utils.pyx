# encoding: utf-8

import numpy as np
cimport numpy as np
cimport cython

from scikits.image.io._plugins.util import prepare_for_display as _prepare_for_display
from PyQt4.Qt import *

def _to_pixmap(darr):
    qimage = QImage(darr.data, darr.shape[1], darr.shape[0],
            darr.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

cdef extern from "c_src/c_level_lines.c":
    int _c_extract_level_lines(float*, int, int, unsigned char*, float, float, int)

def _extract_level_lines(np.ndarray[np.float32_t, ndim=2]im, float ofs, float step, int mode=1):
    cdef int nx = im.shape[1]
    cdef int ny = im.shape[0]

    cdef float fstep = float(step)

    cdef np.ndarray[np.float32_t] flat_im = im.ravel()
    cdef np.ndarray[np.float32_t, mode='c'] contim
    try:
        contim = flat_im
    except:
        contim = flat_im.astype(np.float32)

    cdef np.ndarray[np.uint8_t] out = np.empty(ny*nx, dtype=np.uint8)

    c = _c_extract_level_lines(<float*>contim.data, nx, ny,
            <unsigned char*>out.data, ofs, step, mode)
    if c != 0:
        raise Exception("_c_extract_level_lines exited with return code "+str(c))

    return out.reshape((ny, nx)) > 0

cdef class IntBox(object):
    """
    Integer axis-aligned boxes operations (intersection, difference, etc.)
    This is useful when trying to recompute as few information as possible when
    displaying a new view.
    """
    cdef public np.int32_t x, y, w, h

    def __init__(self, int x, int y, int w, int h):
        """
        x, y, w, h: integers describing the box
        """
        w = max(0, w)
        h = max(0, h)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w*self.h

    def is_empty(self):
        return self.w*self.h == 0

    def shifted(self, int dx, int dy):
        return IntBox(self.x+dx, self.y+dy, self.w, self.h)

    def coords(self):
        return (self.x, self.y, self.w, self.h)

    def copy(self):
        return IntBox(self.x, self.y, self.w, self.h)

    def intersection(self, box not None):
        """
        Returns the intersecting box
        """
        x, y, w, h = self.coords()
        a, b, s, t = box.coords()

        x0 = max(x, a)
        y0 = max(y, b)
        x1 = min(x+w-1, a+s-1)
        y1 = min(y+h-1, b+t-1)

        cw = max(0, x1-x0+1)
        ch = max(0, y1-y0+1)

        return IntBox(x0, y0, cw, ch)

    def difference(self, box not None):
        """
        Returns a list of non-empty boxes whose union is self - box
        """
        r = self
        b0, r = r.split_horiz(box.x-1)
        r, b1 = r.split_horiz(box.x+box.w-1)
        b2, r = r.split_vert(box.y-1)
        r, b3 = r.split_vert(box.y+box.h-1)
        intersection = r

        difference_list = filter(lambda b: not b.is_empty(), [b0,b1,b2,b3])
        return difference_list

    def split_horiz(self, int x):
        """
        Returns a pair of boxes whose union is self, and that are contained
        respectively in the plane (X <= x) or (X > x)
        """
        a, b, s, t = self.coords()
        x0 = max(x+1, a)
        return (IntBox(a, b, min(s, x-a+1), t),
                IntBox(x0, b, a+s-x0, t))

    def split_vert(self, int y):
        """
        Returns a pair of boxes whose union is self, and that are contained
        respectively in the plane (Y <= y) or (Y > y)
        """
        a, b, s, t = self.coords()
        y0 = max(y+1, b)
        return (IntBox(a, b, s, min(t, y-b+1)),
                IntBox(a, y0, s, b+t-y0))

