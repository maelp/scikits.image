#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* extract image value (even outside image domain) */
static inline float
v(float* in, int w, int h, int x, int y, float bg)
{
  if (x<0 || x>=w || y<0 || y>=h)
    return(bg); else return(in[y*w+x]);
}

/* c[] = values of interpolation function at ...,t-2,t-1,t,t+1,... */

/* coefficients for cubic interpolant (Keys' function) */
static void
keys(float *c, float t, float a)
{
  float t2,at;

  t2 = t*t;
  at = a*t;
  c[0] = a*t2*(1.0-t);
  c[1] = (2.0*a+3.0 - (a+2.0)*t)*t2 - at;
  c[2] = ((a+2.0)*t - a-3.0)*t2 + 1.0;
  c[3] = a*(t-2.0)*t2 + at;
}

/* coefficients for cubic spline */
static void
spline3(float *c, float t)
{
  float tmp;

  tmp = 1.-t;
  c[0] = 0.1666666666*t*t*t;
  c[1] = 0.6666666666-0.5*tmp*tmp*(1.+t);
  c[2] = 0.6666666666-0.5*t*t*(2.-t);
  c[3] = 0.1666666666*tmp*tmp*tmp;
}

static double
initcausal(double *c, int n, double z)
{
  double zk,z2k,iz,sum;
  int k;

  zk = z; iz = 1./z;
  z2k = pow(z,(double)n-1.);
  sum = c[0] + z2k * c[n-1];
  z2k = z2k*z2k*iz;
  for (k=1;k<=n-2;k++) {
    sum += (zk+z2k)*c[k];
    zk *= z;
    z2k *= iz;
  }
  return (sum/(1.-zk*zk));
}

static double
initanticausal(double *c, int n, double z)
{
  return((z/(z*z-1.))*(z*c[n-2]+c[n-1]));
}

static void
invspline1D(double *c, int size, double *z, int npoles)
{
  double lambda;
  int n,k;

  /* normalization */
  for (k=npoles,lambda=1.;k--;) lambda *= (1.-z[k])*(1.-1./z[k]);
  for (n=size;n--;) c[n] *= lambda;

  /*----- Loop on poles -----*/
  for (k=0;k<npoles;k++) {

    /* forward recursion */
    c[0] = initcausal(c,size,z[k]);
    for (n=1;n<size;n++) 
      c[n] += z[k]*c[n-1];

    /* backwards recursion */
    c[size-1] = initanticausal(c,size,z[k]);
    for (n=size-1;n--;) 
      c[n] = z[k]*(c[n+1]-c[n]);
    
  }
}

static int
finvspline(float *in, int nx, int ny, int order, float *out)
{
  double *c,*d,z[5];
  int npoles,x,y;
 
  /* initialize poles of associated z-filter */
  switch (order) 
    {
    case 2: z[0]=-0.17157288;  /* sqrt(8)-3 */
      break;

    case 3: z[0]=-0.26794919;  /* sqrt(3)-2 */ 
      break;

    case 4: z[0]=-0.361341; z[1]=-0.0137254;
      break;

    case 5: z[0]=-0.430575; z[1]=-0.0430963;
      break;
      
    case 6: z[0]=-0.488295; z[1]=-0.0816793; z[2]=-0.00141415;
      break;

    case 7: z[0]=-0.53528; z[1]=-0.122555; z[2]=-0.00914869;
      break;
      
    case 8: z[0]=-0.574687; z[1]=-0.163035; z[2]=-0.0236323; z[3]=-0.000153821;
      break;

    case 9: z[0]=-0.607997; z[1]=-0.201751; z[2]=-0.0432226; z[3]=-0.00212131;
      break;
      
    case 10: z[0]=-0.636551; z[1]=-0.238183; z[2]=-0.065727; z[3]=-0.00752819;
      z[4]=-0.0000169828;
      break;
      
    case 11: z[0]=-0.661266; z[1]=-0.27218; z[2]=-0.0897596; z[3]=-0.0166696; 
      z[4]=-0.000510558;
      break;
      
     default:
      /* mwerror(FATAL,1,"finvspline: order should be in 2..11.\n"); */
      return -1;
    }
  npoles = order/2;

  /* initialize double array containing image */
  c = (double *)malloc(nx*ny*sizeof(double));
  d = (double *)malloc(nx*ny*sizeof(double));
  for (x=nx*ny;x--;) 
    c[x] = (double)in[x];

  /* apply filter on lines */
  for (y=0;y<ny;y++) 
    invspline1D(c+y*nx,nx,z,npoles);

  /* transpose */
  for (x=0;x<nx;x++)
    for (y=0;y<ny;y++) 
      d[x*ny+y] = c[y*nx+x];
      
  /* apply filter on columns */
  for (x=0;x<nx;x++) 
    invspline1D(d+x*ny,ny,z,npoles);

  /* transpose directy into image */
  for (x=0;x<nx;x++)
    for (y=0;y<ny;y++) 
      out[y*nx+x] = (float)(d[x*ny+y]);

  /* free array */
  free(d);
  free(c);

  return 0;
}

/* pre-computation for spline of order >3 */
static void
init_splinen(float *a, int n)
{
  int k;

  a[0] = 1.;
  for (k=2;k<=n;k++) a[0]/=(float)k;
  for (k=1;k<=n+1;k++)
    a[k] = - a[k-1] *(float)(n+2-k)/(float)k;
}

/* fast integral power function */
static inline float
ipow(float x, int n)
{
  float res;

  for (res=1.;n;n>>=1) {
    if (n&1) res*=x;
    x*=x;
  }
  return(res);
}

/* coefficients for spline of order >3 */
static void
splinen(float *c, float t, float *a, int n)
{
  int i,k;
  float xn;
  
  memset((void *)c,0,(n+1)*sizeof(float));
  for (k=0;k<=n+1;k++) { 
    xn = ipow(t+(float)k,n);
    for (i=k;i<=n;i++) 
      c[i] += a[i-k]*xn;
  }
}

static int
_c_zoom(
        float *in, /* Input image */
        int nx, int ny, /* Size of the input image */
        float fx, float fy, /* Coordinates (in the input image) of the sample
                               corresponding to the first pixel in the output image */
        float *out, /* Output image */
        int wx, int wy, /* Size of the output image */
        float z, /* Desired zoom */
        float bgcolor, /* Background color */
        int o /* Zoom order */
)
{
  int    n1,n2,x,y,xi,yi,d;
  float  zx,zy,res,xp,yp,u,c[12],ak[13];
  float  *ref,*tmp,*coeffs;

  if( fabs(z) < 1E-4 ) return -1;

  /* We only want homogeneous zooms (for now) */
  zx = z;
  zy = z;

  /* Keys parameter for the bicubic zoom, can be in [-1.0, 0.0] */
  float p = -0.5;

  /* CHECK ORDER */
  if (o!=0 && o!=1 && o!=-3 && o!=3 && o!=5 && o!=7 && o!=9 && o!=11)
      return -1;
  if (wx<0 || wy<0)
      return -1;

  if (o>=3) {
    coeffs = (float*)calloc(nx*ny, sizeof(float));
    if( coeffs == NULL )
        return 1;
    finvspline(in,nx,ny,o,coeffs);
    ref = coeffs;
    if (o>3) init_splinen(ak,o);
  }
  else {
    coeffs = NULL;
    ref = in;
  }

  tmp = (float*)calloc(ny*wx, sizeof(float));
  if( tmp == NULL )
      return 1;

  /********** FIRST LOOP (x) **********/
  
  for (x=0;x<wx;x++)
  {
    xp = fx+( (float)x + 0.5 )/zx;

    if (o==0)
    { /* zero order interpolation (pixel replication) */
      xi = (int)floor((double)xp); 
      if (xi<0 || xi>=nx)
      {
          for (y=0;y<ny;y++) tmp[y*wx+x] = bgcolor; 
      }
      else
      {
          for (y=0;y<ny;y++) tmp[y*wx+x] = ref[y*nx+xi];
      }
    }
    else { /* higher order interpolations */
      if (xp<0. || xp>(float)nx) 
      {
          for (y=0;y<ny;y++) tmp[y*wx+x] = bgcolor; 
      }
      else
      {
        xp -= 0.5;
        xi = (int)floor((double)xp); 
        u = xp-(float)xi;
        switch (o) 
        {
        case 1: /* first order interpolation (bilinear) */
          n2 = 1; c[0]=u; c[1]=1.-u; break;
          
        case -3: /* third order interpolation (bicubic Keys' function) */
          n2 = 2; keys(c,u,p); break;
          
        case 3: /* spline of order 3 */
          n2 = 2; spline3(c,u); break;
          
        default: /* spline of order >3 */
          n2 = (1+o)/2; splinen(c,u,ak,o); break;
        }

        n1 = 1-n2;
        /* this test saves computation time */
        if (xi+n1>=0 && xi+n2<nx) {
          for (y=0;y<ny;y++) {
            for (d=n1,res=0.;d<=n2;d++) 
              res += c[n2-d]*ref[y*nx+xi+d];
            tmp[y*wx+x] = res;
          }
        }
        else 
        {
          for (y=0;y<ny;y++) {
            for (d=n1,res=0.;d<=n2;d++) 
              res += c[n2-d]*v(ref,nx,ny,xi+d,y,bgcolor);
            tmp[y*wx+x] = res;
          }
        }
      }
    }
  }
  
  ref = tmp;

  /********** SECOND LOOP (y) **********/
  
  for (y=0;y<wy;y++)
  {

    yp = fy+( (float)y + 0.5 )/zy;

    if (o==0)
    { /* zero order interpolation (pixel replication) */
      yi = (int)floor((double)yp); 
      if (yi<0 || yi>=ny)
      {
          for (x=0;x<wx;x++) out[y*wx+x] = bgcolor; 
      }
      for (x=0;x<wx;x++) out[y*wx+x] = ref[yi*wx+x];
    }
    else
    { /* higher order interpolations */
      if (yp<0. || yp>(float)ny) 
      {
          for (x=0;x<wx;x++) out[y*wx+x] = bgcolor; 
      }
      else
      {
        yp -= 0.5;
        yi = (int)floor((double)yp); 
        u = yp-(float)yi;
        switch (o) 
        {
        case 1: /* first order interpolation (bilinear) */
          n2 = 1; c[0]=u; c[1]=1.-u; break;
          
        case -3: /* third order interpolation (bicubic Keys' function) */
          n2 = 2; keys(c,u,p); break;
          
        case 3: /* spline of order 3 */
          n2 = 2; spline3(c,u); break;
          
        default: /* spline of order >3 */
          n2 = (1+o)/2; splinen(c,u,ak,o); break;
        }
        
        n1 = 1-n2;
        /* this test saves computation time */
        if (yi+n1>=0 && yi+n2<ny)
        {
          for (x=0;x<wx;x++)
          {
            for (d=n1,res=0.;d<=n2;d++) 
              res += c[n2-d]*ref[(yi+d)*wx+x];
            out[y*wx+x] = res;
          }
        }
        else
        {
          for (x=0;x<wx;x++)
          {
            for (d=n1,res=0.;d<=n2;d++) 
              res += c[n2-d]*v(ref,wx,ny,x,yi+d,bgcolor);
            out[y*wx+x] = res;
          }
        }
      }
    }
  }

  free(tmp);
  if (coeffs) free(coeffs);

  return 0;
}
