#!/bin/bash

image_dir=$1

for i in ${image_dir}/*.JPEG; do file $i | awk 'BEGIN{minh=500;minw=500;}{split($0,a,":"); printf("%s ",a[1]);x=split($0,b,","); split(b[x-1],c,"x"); printf("%d %d %f ",c[1],c[2],c[1]/c[2]); if (c[1] > c[2]) rat=c[1]/c[2]; else rat=c[2]/c[1]; rat=rat-1; if (rat < 0) rat = -rat; printf("%f  big %f\n",c[1]/c[2],rat);}'>> output2; done; sort -n -k 7 output | less
