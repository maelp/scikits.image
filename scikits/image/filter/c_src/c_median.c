#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int
comp(void* i,void* j)
{ 
  return ((int)*((unsigned char *)i)-(int)*((unsigned char *)j));
}

int c_median_iteration(unsigned char* u, int nx, int ny, unsigned char* v, int* s, int ns)
{
  int           changed;
  register int  i,x,y,xx,yy,p;
  unsigned char curr[10000];
  long          adr;
  int           px,py;

  changed = 0;
  for (x=0;x<nx;x++)
  {
    for (y=0;y<ny;y++)
    {
      i=0;
      for (p=0;p<ns;p++)
      {
        px = s[p*2+0];
        py = s[p*2+1];
        xx = x+px;
        yy = y+py;
        if( xx>=0 && xx<nx && yy>=0 && yy<ny )
        {
          curr[i++] = u[yy*nx+xx];
        }
      }
      qsort((char *)curr, i, sizeof(char), comp);
      adr = y*nx+x;
      v[adr] = curr[i/2];
      if( v[adr] != u[adr] ) changed++;
    }
  }
  return changed;
}
