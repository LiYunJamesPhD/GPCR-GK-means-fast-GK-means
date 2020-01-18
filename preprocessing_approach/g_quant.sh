# declare variables
input_image=$1
output_file_path=$2
k_value=$3

user=$(whoami)

rm /home/$user/palette.ppm > /dev/null 2>&1
rm /home/$user/tempBlur.png > /dev/null 2>&1
rm /home/$user/tempBlur.ppm > /dev/null 2>&1
rm /home/$user/tempReduced.ppm > /dev/null 2>&1
rm /home/$user/log.txt > /dev/null 2>&1

echo $1 >> /home/$user/log.txt
python3 gauss.py --input_img $input_image \
                 --output_img /home/$user/tempBlur.png
pngtopnm /home/$user/tempBlur.png > /home/$user/tempBlur.ppm

pnmcolormap $k_value /home/$user/tempBlur.ppm > /home/$user/palette.ppm 2>> /home/$user/log.txt
pnmremap -map=/home/$user/palette.ppm /home/$user/tempBlur.ppm > /home/$user/tempReduced.ppm 2>> /home/$user/log.txt 
pnmtopng /home/$user/tempReduced.ppm > $output_file_path 2>> /home/$user/log.txt

rm /home/$user/palette.ppm > /dev/null 2>&1
rm /home/$user/tempBlur.png > /dev/null 2>&1
rm /home/$user/tempBlur.ppm > /dev/null 2>&1
rm /home/$user/tempReduced.ppm > /dev/null 2>&1
