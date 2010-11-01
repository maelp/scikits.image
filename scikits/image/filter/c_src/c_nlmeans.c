#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

int
c_nlmeans(
        float* in, float* out, int nx0, int ny0, /* input/output buffers */
        float h, /* regularization parameter */
        int s,   /* patch side length (odd integer) */
        float a, /* decay of Euclidean patch distance */
        int d,   /* maximum patch distance */
        float c, /* weight for self-patch */
        unsigned char *mask, int mx, int my /* do not denoise pixels x with mask(x) = 0 */
)
{
  int *dadr=NULL,*dd=NULL,nx,ny,x,y,xp,yp,i,adr,adrp,wsize,ds;
  double *w=NULL,*ww=NULL,dist,new,sum,e,A,*ref=NULL;
  register double v;

  if (s<1 || ((s)&1)==0) 
  {
      /* Parameter error */
      return -1 ;
  }
  A = 2.*a*a; if (A==0.) A=1.;
  ds = (s-1)/2; /* patch = [-ds,ds]x[-ds,ds] */

  nx = nx0+2*ds; 
  ny = ny0+2*ds;

  ref = (double *)malloc(nx*ny*sizeof(double));
  if (!ref)
      goto err_not_enough_memory;

  /* enlarge image to deal properly with borders */
  for (y=0;y<ny;y++) {
    yp = y-ds;
    if (yp<0) yp=-yp;
    if (yp>=ny0) yp=ny0*2-2-yp;
    for (x=0;x<nx;x++) {
      xp = x-ds;
      if (xp<0) xp=-xp;
      if (xp>=nx0) xp=nx0*2-2-xp;
      ref[y*nx+x] = (double)in[yp*nx0+xp];
    }
  }

  /* precompute weights */
  wsize = s*s;
  w = (double *)malloc(wsize*sizeof(double));
  dadr = (int *)malloc(wsize*sizeof(int));
  if( !w || !dadr )
      goto err_not_enough_memory;

  for(sum=0.,i=0,x=-ds;x<=ds;x++)
    for(y=-ds;y<=ds;y++,i++) {
      dadr[i] = y*nx+x;
      w[i] = exp(-(double)(x*x+y*y)/A);
      sum += w[i];
    }
  for (i=wsize;i--;) w[i] /= sum*2.*h*h;

  /* main loop */
  for (x=ds;x<nx-ds;x++)
  {
      /*printf("x=%d/%d\n",x-ds+1,nx-ds*2);*/
      for (y=ds;y<ny-ds;y++)
      {
          adr = y*nx+x;
          if (!mask || mask[adr])
          {
              new = sum = 0.;
              /* loop on patches */
              for (xp=MAX(x-d,ds);xp<=MIN(x+d,nx-1-ds);xp++)
              {
                  for (yp=MAX(y-d,ds);yp<=MIN(y+d,ny-1-ds);yp++)
                  {
                      adrp = yp*nx+xp;
                      for (i=wsize,dist=0.,ww=w,dd=dadr;i--;ww++,dd++)
                      {
                          v = ref[adr+*dd]-ref[adrp+*dd];
                          dist += *ww*v*v;
                      }
                      e = (adrp==adr?c:exp(-dist));
                      new += e*(double)ref[adrp];
                      sum += e;
                  }
              }
              out[(y-ds)*nx0+x-ds] = (float)(new/sum);
          }
      }
  }
  free(ref);
  free(dadr);
  free(w);
  return 0;

err_not_enough_memory:
  free(ref);
  free(dadr);
  free(w);
  return 1;
}
