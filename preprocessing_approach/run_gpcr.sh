# declare variables
dataset_folder_path=$1
k_value=$2  # E.g., 64 128 256 512
destination_folder_path=$3

mkdir $destination_folder_path
mkdir $destination_folder_path/images

# =============================================
for file_path in $dataset_folder_path/*.png; do
    file_name=$(basename $file_path .png)
    new_file_name=$file_name.gpcr.$k_value.png

    # perform Gaussian and kmeans
    ./g_quant.sh $file_path $destination_folder_path/images/$new_file_name $k_value
done


