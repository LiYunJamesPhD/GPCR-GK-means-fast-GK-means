#!/bin/bash

# declare variables
dataset_folder_path=$1
model=$2  # E.g., iv3 (Inception_v3)
bit_num=$3  # E.g., 1 to 6
destination_path=$4

mkdir $dataset_folder_path/bit_reduction
mkdir $dataset_folder_path/bit_reduction/images

for adv_img_path in $dataset_folder_path/*.png; do
    file_name=$(basename $adv_img_path)
    new_file_name="${file_name%.*}.bit_reduc.$bit_num"

    # convert a png image to a ppm image
    pngtopnm $adv_img_path > $destination_path/${file_name%.*}.ppm

    # run bit reduction
    ./bit_reduction $destination_path/${file_name%.*}.ppm -n $bit_num > $destination_path/$new_file_name.ppm
    
    # convert the ppm image back to a png image
    pnmtopng -force $destination_path/$new_file_name.ppm > $destination_path/bit_reduction/images/$new_file_name.png

    rm $destination_path/*.ppm
done


