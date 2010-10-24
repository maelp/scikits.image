#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void
_c_shock_iter(float* u, float* v, int width, int height, float s)
{
  int   i,j,im,i1,jm,j1;
  float new,val;
  float laplacian;

  for (i=0;i<width;i++) 
    for (j=0;j<height;j++) {

      if (j==0) jm=1; else jm=j-1;
      if (j==height-1) j1=height-2; else j1=j+1;
      if (i==0) im=1; else im=i-1;
      if (i==width-1) i1=width-2; else i1=i+1;

      laplacian=(
             u[width * j  + i1]+
             u[width * j  + im]+
             u[width * j1 + i ]+
             u[width * jm + i ]-
	     4.0*u[width * j  + i]);
      
      new = u[width * j  + i];

      if (laplacian > 0.0)
      {
        /* erosion */
        val = u[width * j  + i1]; if (val<new) new = val;
        val = u[width * j  + im]; if (val<new) new = val;
        val = u[width * j1 + i ]; if (val<new) new = val;
        val = u[width * jm + i ]; if (val<new) new = val;
      } else if (laplacian < 0.0)
      {
        /* dilation */
        val = u[width * j  + i1]; if (val>new) new = val;
        val = u[width * j  + im]; if (val>new) new = val;
        val = u[width * j1 + i ]; if (val>new) new = val;
        val = u[width * jm + i ]; if (val>new) new = val;
      }

      v[width*j+i] = s * new + (1.0-s) * u[width*j+i];
    }
}

int
c_shock(float* u, float* v, int nx, int ny, int n, float s)
{
  int    i;
  float *old, *new, *tmp;

  old = u;
  new = v;

  for( i=0 ; i<n ; i++ )
  {
    _c_shock_iter(old,new,nx,ny,s);
    tmp=old; old=new; new=tmp;
  }
  
  if (old == u)
  {
    for( i=0 ; i < nx*ny ; i++ )
    {
      *v++ = *u++ ;
    }
  }

  return 0;
}
