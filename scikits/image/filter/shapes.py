# encoding: utf-8

import numpy as np

# (0, 0) is the center of the first pixel
# a pixel is in the disc if its center is in the continuous ball

def disc(radius):
  assert(radius >= 0)
  c = int(np.floor(radius))
  width = 2*c+1
  mask = np.zeros((width,width), dtype=np.int8)
  rad_sq = radius*radius
  sqrt = np.sqrt
  for x in xrange(c+1):
    y = int(np.floor(sqrt(rad_sq - x*x)))
    mask[c+x,c-y:c+y+1] = 1
    mask[c-x,c-y:c+y+1] = 1
  return mask

def check_disc():
  assert disc(0) == np.array([[1]], dtype=np.int8)
  assert disc(0.9) == np.array([[1]], dtype=np.int8)
  assert disc(1.0) == np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=np.int8)
  assert disc(2.8) == np.array([[0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0]], dtype=np.int8)

def circle(radius):
  assert(radius >= 0)
  c = int(np.floor(radius))
  width = 2*c+1
  mask = np.zeros((width,width), dtype=np.int8)
  rad_sq = radius*radius
  sqrt = np.sqrt
  for x in xrange(c):
    y = int(np.floor(sqrt(rad_sq - x*x)))
    mask[c+x,c+y] = 1
    mask[c+x,c-y] = 1
    mask[c-x,c+y] = 1
    mask[c-x,c-y] = 1
  # Frontiers
  yc = int(np.floor(sqrt(rad_sq - c*c)))
  mask[2*c,c-yc:c+yc+1] = 1
  mask[0,c-yc:c+yc+1] = 1
  return mask

def check_circle():
  assert circle(0) == np.array([[1]], dtype=np.int8)
  assert circle(0.9) == np.array([[1]], dtype=np.int8)
  assert circle(1.0) == np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]], dtype=np.int8)
  assert circle(2.8) == np.array([[0, 1, 1, 1, 0],
                                  [1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1],
                                  [0, 1, 1, 1, 0]], dtype=np.int8)

def shape_coords(shape):
  n_coords = (shape>0).sum()
  coords = np.zeros((n_coords,2), dtype=np.int32)
  nx, ny = shape.shape
  cx, cy = nx/2, ny/2
  n_coord = 0
  for x in xrange(nx):
    for y in xrange(ny):
      if shape[x,y] > 0:
        coords[n_coord][0] = x-cx
        coords[n_coord][1] = y-cy
        n_coord += 1
  return coords
