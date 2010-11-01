#include <stdio.h>
#include <math.h>

int _c_extract_level_lines(
        float *in, /* Input image */
        int nx, int ny, /* Size of the input image */
        unsigned char *out, /* Output image */
        float ofs, float step,
        int mode
)
{
    memset(out, 0, nx*ny);

    float v,w;
    int x,y,adr,ok;
    double fv;

    for (x=0;x<nx-1;x++)
        for (y=0;y<ny-1;y++) {
            adr = y*nx+x;
            v = in[adr];
            ok = 0;

            switch(mode) 
            {
                case 1: /* level lines */
                    fv = floor((double)((v-ofs)/step));
                    w = in[adr+1];
                    if (floor((double)((w-ofs)/step)) != fv) ok = 1;
                    w = in[adr+nx];
                    if (floor((double)((w-ofs)/step)) != fv) ok = 1;
                    w = in[adr+nx+1];
                    if (floor((double)((w-ofs)/step)) != fv) ok = 1;
                    break;

                case 2: /* one level line */
                    w = in[adr+1];
                    if ((w-ofs)*(v-ofs)<=0. && v!=w) ok=1;
                    w = in[adr+nx];
                    if ((w-ofs)*(v-ofs)<=0. && v!=w) ok=1;
                    w = in[adr+nx+1];
                    if ((w-ofs)*(v-ofs)<=0. && v!=w) ok=1;
                    break;

                case 3: /* one lower level set */
                    ok = (v<ofs);
                    break;

                case 4: /* one upper level set */
                    ok = (v>=ofs);
                    break;

                case 5: /* one bi-level set */
                    ok = (v>=ofs && v<ofs+step);
                    break;

            }
            if (ok) {
                out[adr] = 255;
            }
        }

    return 0;
}
