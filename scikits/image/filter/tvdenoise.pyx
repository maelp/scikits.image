import numpy as np
cimport numpy as np

cdef extern from "c_src/c_tvdenoise.c":
  int c_tvdenoise_float(float *, int, int, int, float, float *)
  int c_tvdenoise_double(double *, int, int, int, double, double *)

def tvdenoise_float32(arr, int n=10, float W=50.0):
  """
  Perform total-variation denoising on a float32 array

  Parameters
  ----------
  arr: ndarray
       input data to be denoised

  n: int, optional
     number of iterations

  W: float, optional
     denoising weight
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

  err_code = c_tvdenoise_float(<float *>contarr.data, nx, ny, n, W, <float *>out.data)
  if err_code != 0:
    raise RuntimeError('c_tvdenoise exited with code '+str(err_code))

  return out.reshape((nx, ny))

def tvdenoise_float64(arr, int n=10, double W=50.0):
  """
  Perform total-variation denoising on a float64 array

  Parameters
  ----------
  arr: ndarray
       input data to be denoised

  n: int, optional
     number of iterations

  W: float, optional
     denoising weight
  """
  if len(arr.shape) != 2:
    raise ValueError('Expecting a 2d array')
  cdef int nx = arr.shape[0]
  cdef int ny = arr.shape[1]

  cdef np.ndarray[np.float64_t] flat_arr = arr.ravel()
  cdef np.ndarray[np.float64_t, mode='c'] contarr
  try:
      contarr = flat_arr
  except:
      contarr = flat_arr.astype(np.float64)
  cdef np.ndarray[np.float64_t] out = np.zeros_like(contarr)

  err_code = c_tvdenoise_double(<double *>contarr.data, nx, ny, n, W, <double *>out.data)
  if err_code != 0:
    raise RuntimeError('c_tvdenoise exited with code '+str(err_code))

  return out.reshape((nx, ny))
