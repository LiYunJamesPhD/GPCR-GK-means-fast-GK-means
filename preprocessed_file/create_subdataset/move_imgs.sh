#!/bin/bash

img_list_file=$1
folder_path=/home/li-yun/sub_dataset
mkdir $folder_path
mkdir $folder_path/images

# read a file line by line.
while read -r row; do
    echo "line: " $row
    # copy files to a directory.
    cp /home/li-yun/Desktop/squared_imgs_5000/images/$row $folder_path/images
done < $img_list_file

