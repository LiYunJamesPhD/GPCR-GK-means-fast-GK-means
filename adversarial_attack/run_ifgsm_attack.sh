#!/bin/bash

# declare variables
selected_img_folder=$1
adv_model=$2  # E.g., [resnet18, resnet50, inception_v3]
adversarial_folder=$3

Eps_val=16   # can change to other values from 0 to 255
adv_sub_folder="adv-img-folder"
per_sub_folder="per-img-folder"
log_sub_folder="log-file-folder"
orig_sub_folder="orig-img-folder"
images_folder="images"

# stage 2: adversarial-image generation
# input: the directory with selected images
# output: a directory including adversarial images, log files, and perturbation images (ex: output/images)
if [[ ! -d $adversarial_folder ]]; then
    mkdir $adversarial_folder
else
    echo "$adversarial_folder has been created !!"
fi

if [[ ! -d "$adversarial_folder/$adv_sub_folder" ]]; then
    mkdir "$adversarial_folder/$adv_sub_folder"
else
    echo adv_img directory has been created !!
fi
if [[ ! -d "$adversarial_folder/$adv_sub_folder/$images_folder" ]]; then
    mkdir "$adversarial_folder/$adv_sub_folder/$images_folder"
else
    echo images directory has been created !!
fi
if [[ ! -d "$adversarial_folder/$per_sub_folder" ]]; then
    mkdir "$adversarial_folder/$per_sub_folder"
else
    echo per_img directory has been created !!
fi
if [[ ! -d "$adversarial_folder/$log_sub_folder" ]]; then
    mkdir "$adversarial_folder/$log_sub_folder"
else
    echo log_file directory has been created !!
fi
if [[ ! -d "$adversarial_folder/$orig_sub_folder" ]]; then
    mkdir $adversarial_folder/$orig_sub_folder
else
    echo orig-img directory has bee created !!
fi

max_iter_num=30

# loop through all images
for adv_img_path in $selected_img_folder/*; do
    image_name=$(basename $adv_img_path)

    cd /home/li-yun/ism_github/adversarial_attack/ifgsm

    python3 iterative_fgsm.py --img $adv_img_path --model $adv_model --eps $Eps_val \
    --itr $max_iter_num --save --out_folder $adversarial_folder \
    >>"$adversarial_folder/$log_sub_folder/$image_name-log_$(date +"%Y_%m_%d_%I_%M_%p")"

    cd /home/li-yun/ism_github/adversarial_attack/
done


