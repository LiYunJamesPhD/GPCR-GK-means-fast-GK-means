import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import cv2
import os, sys
#import pretrainedmodels
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# declare variables
in_img_path = sys.argv[1]
model_name = sys.argv[2]
overshoot_val = float(sys.argv[3])
out_folder_path = sys.argv[4]

print('Name: ', model_name)
short_name = ''
if model_name == 'resnet18':
    short_name = 'r18'
elif model_name == 'resnet50':
    short_name = 'r50'
elif model_name == 'inception_v3':
    short_name = 'iv3'
elif model_name == 'inception_v4':
    short_name = 'iv4'
    print('short name:', short_name)

out_num = 0.0
if overshoot_val == 0.02:
    out_num = 2
out_file_path = os.path.join(out_folder_path,
                             os.path.splitext(os.path.basename(in_img_path))[0] + 
                             '.' + short_name + '.dfool.eta' + str(out_num).zfill(2) + '.png')
orig_file_path = os.path.join(out_folder_path,
                              os.path.splitext(os.path.basename(in_img_path))[0] +
                              '.' + short_name + '.preprocessed.png')

net = None
if model_name == 'inception_v4':
    model_name = 'inceptionv4'
    net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
else:
    net = getattr(models, model_name)(pretrained=True)

# Switch to evaluation mode
net.eval()

im_orig = Image.open(in_img_path).convert('RGB')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean
size = 0
if 'inception' in model_name:
    size = 299
else:
    size = 224
origs = transforms.Compose([
        transforms.Resize(299),  # orig: 256
        transforms.CenterCrop(size)])(im_orig)
im = transforms.Compose([
    transforms.Resize(299),  # orig: 256
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net, num_classes = 1000, overshoot = overshoot_val)

print('Loop i:', loop_i)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                         transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                         transforms.Lambda(clip)])

numpy_img_orig = np.asarray(origs)
numpy_img_orig = numpy_img_orig[..., ::-1] # RGB to BGR
cv2.imwrite(orig_file_path, numpy_img_orig)

numpy_img = tf(pert_image.data.cpu()[0]).numpy().transpose(1, 2, 0)
numpy_img = numpy_img * 255.0
numpy_img = numpy_img[..., ::-1] # RGB to BGR
numpy_img = np.clip(numpy_img, 0, 255).astype(np.uint8)
cv2.imwrite(out_file_path, numpy_img)


