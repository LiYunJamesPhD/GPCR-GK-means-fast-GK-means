# declare variables
dataset_folder_path=$1
k_value=$2
destination_folder_path=$3

# =============================================
for file_path in $dataset_folder_path/*.png; do
    file_name=$(basename $file_path .png)
    new_file_name=$file_name.fast_gk.$k_value.png

    # perform Gaussian and kmeans
    ./g_kmean_faster.sh $file_path $destination_folder_path/$new_file_name $k_value
done

