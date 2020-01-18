# declare variables
dataset_folder_path=$1
k_value=$2
destination_folder_path=$3

# =============================================
image_path=$dataset_folder_path/*.png

for file_path in $image_path; do
    #echo $file_path
    file_name=$(basename $file_path .png)
    new_file_name=$file_name.gk.$k_value.png

    # perform Gaussian and kmeans
    python3 g_kmeans.py --input_img $file_path \
                     --output_img $destination_folder_path/$new_file_name \
                     --K $k_value
done

