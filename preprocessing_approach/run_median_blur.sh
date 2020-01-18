#!/bin/bash

input_path=$1
output_path=$2
kerne_size=$3

image_path=$input_path/*.png

for img in $image_path; do
    file_name=$(basename $img)
    new_file_name="${file_name%.*}.mb.$kerne_size.png"

    # run median filter
    convert $img -median ${kerne_size}x${kerne_size} $output_path/$new_file_name

done

