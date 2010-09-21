#!/usr/bin/env python
# encoding: utf-8

import numpy as np

from scikits.image import data_dir
from scikits.image.io import *
from scikits.image.filter import tvdenoise

import sys
import os.path
import argparse

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Total-variation denoising')
    parser.add_argument('filename_in', metavar='in', help='the input file')
    parser.add_argument('-W', default=50.0, type=float,
        help='weight on regularization term')
    parser.add_argument('-n', default=10, type=int,
        help='number of iterations')
    args = parser.parse_args()
    
    filename = args.filename_in
    n = args.n
    W = args.W
else:
    filename = os.path.join(data_dir, 'lena256.tif')
    n = 10
    W = 50.0

im = imread(filename)
imshow(im)
im = np.array(im, dtype=np.float32)
im2 = tvdenoise(im, n, W)

def view_float(im):
    m = np.min(im)
    M = np.max(im)
    scale = float(M - m)
    rescaled = np.rint(255.0 * (im - m) / scale)
    imshow(np.array(rescaled, dtype=np.uint8))

view_float(im2)
