#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define C_TVDENOISE(_type) \
int \
c_tvdenoise_ ## _type (_type* A, int nx, int ny, int n, _type weight, _type* u) \
{ \
  _type *px = calloc(nx*ny, sizeof(_type)); \
  _type *py = calloc(nx*ny, sizeof(_type)); \
\
  int k, x, y, addr; \
  _type d, gx, gy, norm; \
\
  if( !(px && py) ) \
  { \
    if( px ) free(px); \
    if( py ) free(py); \
    return -1; \
  } \
\
  for( k = 0 ; k < n ; k++ ) \
  { \
    for( y = 0, addr = 0 ; y < ny ; y++ ) \
    { \
      for( x = 0 ; x < nx ; x++, addr++ ) \
      { \
        d = -px[addr]-py[addr]; \
        if( x > 0 ) d += px[addr-1]; \
        if( y > 0 ) d += py[addr-nx]; \
        u[addr] = A[addr]+d; \
      } \
    } \
    for( y = 0, addr = 0 ; y < ny ; y++ ) \
    { \
      for( x = 0 ; x < nx ; x++, addr++ ) \
      { \
        gx = (x < nx-1)? u[addr+1]-u[addr] : 0.0; \
        gy = (y < ny-1)? u[addr+nx]-u[addr] : 0.0; \
        norm = sqrt(gx*gx+gy*gy); \
        norm = 1.0 + 0.5*norm/weight; \
        px[addr] = (px[addr]-0.25*gx)/norm; \
        py[addr] = (py[addr]-0.25*gy)/norm; \
      } \
    } \
  } \
\
  free(px); free(py); \
\
  return 0; \
}

C_TVDENOISE(float)
C_TVDENOISE(double)

/*
int
c_tvdenoise_double(double* A, int nx, int ny, int n, double weight, double* u)
{
  double *px = calloc(nx*ny, sizeof(double));
  double *py = calloc(nx*ny, sizeof(double));

  if( !(px && py) )
  {
    if( px ) free(px);
    if( py ) free(py);
    return -1;
  }

  for( int k = 0 ; k < n ; k++ )
  {
    for( int y = 0, addr = 0 ; y < ny ; y++ )
    {
      for( int x = 0 ; x < nx ; x++, addr++ )
      {
        double d = -px[addr]-py[addr];
        if( x > 0 ) d += px[addr-1];
        if( y > 0 ) d += py[addr-nx];
        u[addr] = A[addr]+d;
        E += d*d;
      }
    }
    for( int y = 0, addr = 0 ; y < ny ; y++ )
    {
      for( int x = 0 ; x < nx ; x++, addr++ )
      {
        double gx = (x < nx-1)? u[addr+1]-u[addr] : 0.0;
        double gy = (y < ny-1)? u[addr+nx]-u[addr] : 0.0;
        double norm = sqrt(gx*gx+gy*gy);
        norm = 1.0 + 0.5*norm/weight;
        px[addr] = (px[addr]-0.25*gx)/norm;
        py[addr] = (py[addr]-0.25*gy)/norm;
      }
    }
  }

  free(px); free(py);

  return 0;
}
*/
