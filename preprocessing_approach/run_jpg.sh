#!/bin/bash

# declare variables
dataset_folder_path=$1
quality_level=$2
destination_folder_path=$3

image_path=$dataset_folder_path/*.png
for adv_img_path in $image_path; do
    file_name=$(basename $adv_img_path)
    filename_no_ext="${file_name%.*}"
    new_file_name="$filename_no_ext.jp.$quality_level.jpg"
     
    convert $adv_img_path -quality $quality_level $destination_folder_path/$new_file_name
done

