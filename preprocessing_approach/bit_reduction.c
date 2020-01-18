#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 600000

//------------------------------------------------------------------------
int writeppm(unsigned char *buffer, int xsize, int ysize)
{
    FILE *fp;
    int rc;

    //fprintf(stderr, "X_size, Y_size: %d, %d\n", xsize, ysize);
    fprintf(stdout,"P6\n%d %d\n255\n",xsize,ysize);
    rc = fwrite(buffer, xsize * ysize, 3, stdout);
    //fprintf(stderr, "totoal pixel:\n %d ", rc);
    return(rc);
}

//------------------------------------------------------------------------
int readppm(char *filename, unsigned char *buffer, int *xsize, int *ysize)
{
FILE *fp;
int tmp_x, tmp_y;
int depth;

  fp = fopen(filename, "rb");

  // check magic cookie...
  fgets(buffer,200,fp);
  if ((buffer[0]!='P')&&(buffer[1]!='6')){
     fprintf(stderr,"Wrong file format %s\n",filename);
     exit(-1);
     }
  fscanf(fp,"%d",&tmp_x);
  *xsize = tmp_x;
  fscanf(fp,"%d",&tmp_y);
  *ysize = tmp_y;

  fscanf(fp,"%d\n",&depth);
  fprintf(stderr," x %d   y %d  depth %d\n",tmp_x,tmp_y,depth);

  if (tmp_x*tmp_y*3>=MAX){
    fprintf(stderr,"bad array %d too small\n",MAX);
    exit(1);
    }

  int xx = fread(buffer, tmp_x*tmp_y, 3, fp);
  return(tmp_x*tmp_y*3);
}

int main(int argc, char *argv[])
{
    int args = 4;
    int bit_num = 0;
    if ((argv[2][0] == '-') && (argv[2][1] == 'n')){
        bit_num = atoi(argv[3]);
        fprintf(stderr, " desired number of bits!! \n");
    }
    if (argc < args){
        printf("usage: %s file1 -n #_of_bits\n",argv[0]);
        printf("  outputs to stdout\n");
        exit(-1);
    }
    // create a mask for bit-depth reduction
    int pow_val = pow(2, 8 - bit_num);
    int mask = 255 - (pow_val - 1);

    int size1;
    unsigned char img1[MAX];
    short res[MAX];    

    int xsize1, ysize1;
    fprintf(stderr,"Reading %s: ",argv[1]);
    int img_size = readppm(argv[1], img1, &xsize1, &ysize1);

    int i;
    for (i = 0 ; i < img_size ; i++){
        // fprintf(stderr,"before num: %d ", img1[i]);
        int tmp_val = (int)img1[i];
        res[i] = mask & tmp_val;
        img1[i] = (unsigned char)res[i];    
        // fprintf(stderr,"after num: %d ", img1[i]);
    }

    writeppm(img1, xsize1, ysize1);
    return 0;
}


