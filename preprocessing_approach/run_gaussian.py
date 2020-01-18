import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type = str, default='', help='path to input folder')
parser.add_argument('--output_folder', type = str, default='', help='path to output folder')
parser.add_argument('--sd', type = str, default='', help='standard derivation')

args = parser.parse_args()
in_folder_path = args.input_folder
out_folder_path = args.output_folder
sd_value = int(args.sd)

file_list = os.listdir(in_folder_path)

for single_file in file_list:
    new_image_file = os.path.splitext(os.path.basename(single_file))[0] + '.gb.' + str(sd_value) + '.png'
    img = cv2.imread(os.path.join(in_folder_path, single_file))

    # Gaussian filter with kernel size int(2 * sd) + 1
    k_size = int(2 * sd_value) + 1
    blur = cv2.GaussianBlur(img, (k_size, k_size), sd_value)
    
    cv2.imwrite(os.path.join(out_folder_path, new_image_file), blur)


