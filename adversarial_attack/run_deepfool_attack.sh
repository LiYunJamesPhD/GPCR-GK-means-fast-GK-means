#!/bin/bash

# declare variables
selected_img_folder=$1
deep_learning_model=$2
adversarial_folder=$3
step_size=0.02
orig_folder="orig-img-folder"

if [[ ! -d $adversarial_folder ]]; then
    mkdir $adversarial_folder
else
    echo "$adversarial_folder has been created !!"
fi

mkdir "$adversarial_folder/$orig_folder"
mkdir "$adversarial_folder/adv-img-folder"
mkdir "$adversarial_folder/adv-img-folder/images"

# loop through all images
for adv_img_path in $selected_img_folder/*; do
    image_name=$(basename $adv_img_path)

    cd /home/li-yun/ism_github/adversarial_attack/deepfool
        python3 test_deepfool.py $adv_img_path $deep_learning_model $step_size \
        $adversarial_folder
    cd ..
done

# loop through adversarial images inside the directory
cd $adversarial_folder
for sing_orig_img_file in $(ls | grep .preprocessed.png); do
    mv $sing_orig_img_file $orig_folder
done
# loop through adversarial images inside the directory
cd $adversarial_folder
for sing_adv_img_file in $(ls | grep .png); do
    mv $sing_adv_img_file adv-img-folder/images
done
cd ..



