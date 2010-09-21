import numpy as np
cimport numpy as np

cdef extern from "c_src/c_tvdenoise.c":
  int c_tvdenoise_float(float *, int, int, int, float, float *)

def tvdenoise(arr, int niter=10, float W=50.0):
  """
  Perform total-variation denoising on a float32 array

  Parameters
  ----------
  arr: ndarray
       input data to be denoised

  niter: int, optional
     number of iterations

  W: float, optional
     denoising weight

  Returns
  -------
  out : ndarray
    The denoised image
  
  Examples
  --------
  >>> import os
  >>> from scikits.image import data_dir
  >>> from scikits.image.io import imread

  >>> lena = imread(os.path.join(data_dir, 'lena256.tif'))
  >>> lena_denoised = tvdenoise(lena, niter=5)
  """
  if len(arr.shape) != 2:
    raise ValueError('Expecting a 2d array')
  cdef int nx = arr.shape[0]
  cdef int ny = arr.shape[1]

  cdef np.ndarray[np.float32_t] flat_arr = arr.ravel()
  cdef np.ndarray[np.float32_t, mode='c'] contarr
  try:
      contarr = flat_arr
  except:
      contarr = flat_arr.astype(np.float32)
  cdef np.ndarray[np.float32_t] out = np.zeros_like(contarr)

  err_code = c_tvdenoise_float(<float *>contarr.data, nx, ny,
      niter, W, <float *>out.data)
  if err_code != 0:
    raise RuntimeError('c_tvdenoise exited with code '+str(err_code))

  return out.reshape((nx, ny))
