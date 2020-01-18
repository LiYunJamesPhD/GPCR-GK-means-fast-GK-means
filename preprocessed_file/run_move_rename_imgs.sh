#!/bin/bash

# two arguments: $1: 1000_images_list  $2: a path to an output directory
folder_path=$2
label_file_path=/home/li-yun/ism_github/preprocessed_file/val.txt
mkdir $folder_path
mkdir $folder_path/images

# read a file line by line.
while read -r row; do
    echo "line: " $row

    # copy images to a directory.
    cp /home/li-yun/imagenet_dataset/imagenet_val/$row $folder_path/images

    # re-name all the image files
    python3 ~/ism_github/preprocessed_file/rename_imagenet.py $folder_path/images/$row $label_file_path
done < $1



