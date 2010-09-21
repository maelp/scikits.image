#!/usr/bin/env python
# encoding: utf-8

from scikits.image import data_dir
from scikits.image.io import *
from scikits.image.filter import median

import sys
import os.path
import argparse

if len(sys.argv) > 1:
  parser = argparse.ArgumentParser(description='Total-variation denoising')
  parser.add_argument('filename_in', metavar='in', help='the input file')
  parser.add_argument('-r', default=1.0, type=float, help='radius of the disk')
  parser.add_argument('-n', default=10, type=int,
    help='number of iterations')
  args = parser.parse_args()
  
  filename = args.filename_in
  n = args.n
  r = args.r
else:
  filename = os.path.join(data_dir, 'lena256.tif')
  n = 10
  r = 1.0

im=imread(filename)
imshow(im)
im2=median(im, radius=r, niter=n)
imshow(im2)
