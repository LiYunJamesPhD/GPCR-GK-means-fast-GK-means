import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', type = str, default='', help='path to input image')
parser.add_argument('--output_img', type = str, default='', help='path to output image')

args = parser.parse_args()
in_path = args.input_img
out_path = args.output_img

img = cv2.imread(in_path)
# Gaussian filter with kernel size 5
sd_value = 2.0
k_size = int(2 * sd_value) + 1

blur = cv2.GaussianBlur(img, (k_size, k_size), sd_value)

cv2.imwrite(out_path, blur)

