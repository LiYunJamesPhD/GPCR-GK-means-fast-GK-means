# declare variables
adversarial_folder_path=$1
classification_model=$2  # [resnet18, resnet50, inception_v3
pre_process_img=$3   # using pre-processed images: [yes, no]
file_path=$4

base_path=/home/li-yun/ism_github/image_classification/img_classification

cd $base_path

# run input images on DCNN models to get probabilities to all classes
if [[ $pre_process_img == "yes" ]]; then
    python3 main.py --data dataset_imagenet_1000 --test --loadPreprocessImg --evalf $adversarial_folder_path --pretrained \
	    -a $classification_model > $file_path
elif [[ $pre_process_img == "no" ]]; then
    python3 main.py --data dataset_imagenet_1000 --test --evalf $adversarial_folder_path --pretrained \
	    -a $classification_model > $file_path
fi

cd ../..

