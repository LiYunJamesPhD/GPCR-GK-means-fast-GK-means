import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', type = str, default='', help='path to input image')
parser.add_argument('--output_img', type = str, default='', help='path to output image')
parser.add_argument('--K', type = str, default='', help='central number')

args = parser.parse_args()
img_path = args.input_img
out_path = args.output_img
k = int(args.K)

img = cv2.imread(img_path)
sd_value = 2.0
k_size = int(2 * sd_value) + 1

blur = cv2.GaussianBlur(img, (k_size, k_size), sd_value)
new_img = blur.reshape((-1, 3))

# K-means
new_img = np.float32(new_img)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
ret, label, center = cv2.kmeans(new_img, k, None, criteria, 200, cv2.KMEANS_RANDOM_CENTERS)

new_img2 = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite(out_path, res2)

