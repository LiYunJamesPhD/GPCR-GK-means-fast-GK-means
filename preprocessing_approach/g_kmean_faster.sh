# declare variables
input_image=$1
output_file_path=$2
k_value=$3
ksize=$4

user=$(whoami)

rm /home/$user/tempSmaller.png > /dev/null 2>&1
rm /home/$user/tempSmallGk.png > /dev/null 2>&1
rm /home/$user/temp.ppm > /dev/null 2>&1
rm /home/$user/palette.ppm > /dev/null 2>&1
rm /home/$user/tempBlur.png > /dev/null 2>&1
rm /home/$user/tempBlur.ppm > /dev/null 2>&1
rm /home/$user/tempReduced.ppm > /dev/null 2>&1

convert $input_image -resize "150x150" /home/$user/tempSmaller.png
python3 g_kmean.py --input_img /home/$user/tempSmaller.png \
                 --output_img /home/$user/tempSmallGk.png \
                 --K $k_value

pngtopnm /home/$user/tempSmallGk.png > /home/$user/temp.ppm
python3 gauss.py --input_img $input_image \
                 --output_img /home/$user/tempBlur.png
pngtopnm /home/$user/tempBlur.png > /home/$user/tempBlur.ppm

pnmcolormap all /home/$user/temp.ppm > /home/$user/palette.ppm 2> /dev/null
pnmremap -map=/home/$user/palette.ppm /home/$user/tempBlur.ppm > /home/$user/tempReduced.ppm 2> /dev/null
pnmtopng /home/$user/tempReduced.ppm > $output_file_path 2> /dev/null

rm /home/$user/tempSmaller.png > /dev/null 2>&1
rm /home/$user/tempSmallGk.png > /dev/null 2>&1
rm /home/$user/temp.ppm > /dev/null 2>&1
rm /home/$user/palette.ppm > /dev/null 2>&1
rm /home/$user/tempBlur.png > /dev/null 2>&1
rm /home/$user/tempBlur.ppm > /dev/null 2>&1
rm /home/$user/tempReduced.ppm > /dev/null 2>&1
