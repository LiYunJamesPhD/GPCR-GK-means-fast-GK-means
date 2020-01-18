""" Basic Iterative Method (Targeted and Non-targeted)
	Paper link: https://arxiv.org/abs/1607.02533

	Controls: 
		'esc' - exit
		 's'  - save adversarial image
	  'space' - pause
"""
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import cv2
import argparse
from imagenet_classes import classes
from PIL import Image

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A
clip = lambda x: clip_tensor(x, 0, 255)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/goldfish.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18',
					 choices=['resnet18', 'inception_v3', 'resnet50', 'inception_v4'],
					 required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--y_target', type=int, required=False, default=None, help='target label')
parser.add_argument('--eps', type=int)
parser.add_argument('--itr', type=int)
parser.add_argument('--save', action='store_true',
                    help='save two images')
parser.add_argument('--out_folder', type=str, default='', help='the output folder path')

args = parser.parse_args()
image_path = args.img
model_name = args.model
y_true = args.y
y_target = args.y_target
file_name = os.path.basename(image_path)
output_folder = args.out_folder

print('Iterative Method')
print('Model: %s' %(model_name))
print()

# load images and convert them to RGB
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
im_orig = Image.open(image_path).convert('RGB')

size = 0
if model_name == "inception_v3" or model_name == "inception_v4":
    size = 299
else:
    size = 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

orig= transforms.Compose([
        transforms.Resize(299),  # orig: 256
        transforms.CenterCrop(size)])(im_orig)
img= transforms.Compose([
        transforms.Resize(299),  # orig: 256
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)])(im_orig)

short_name = ''
if model_name == 'resnet18':
	short_name = 'r18'
elif model_name == 'resnet50':
	short_name = 'r50'
elif model_name == 'inception_v3':
	short_name = 'iv3'
elif model_name == 'inception_v4':
	short_name = 'iv4'

# save preprocessed images
numpy_img_orig = np.asarray(orig)
numpy_img_orig = numpy_img_orig[..., ::-1] # RGB to BGR
orig_folder = os.path.join(output_folder, 'orig-img-folder')
cv2.imwrite(os.path.join(orig_folder, os.path.splitext(file_name)[0] + '.' + short_name + '.preprocessed' + '.png'),
            numpy_img_orig)

# load model
model = None
if model_name == 'inception_v4':
    model_name = 'inceptionv4'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
else:
    model = getattr(models, model_name)(pretrained=True)

model.eval().cuda()
criterion = nn.CrossEntropyLoss().cuda()

# prediction before attack
orig = Variable(img.cuda().float().unsqueeze(0), requires_grad = True)
out = model(orig)
pred = np.argmax(out.data.cpu().numpy())

if y_target is not None:
	pred = y_target

print('Prediction before attack: %s' %(classes[pred].split(',')[0]))

inp = Variable(img.cuda().float().unsqueeze(0), requires_grad=True)
print('input data:', inp.data.cpu().numpy().shape)
eps = args.eps
alpha = 1
num_iter = args.itr
print('eps [%d]' %(eps))
print('Iter [%d]' %(num_iter))
print('alpha [1]')
print('-'*20)

for i in range(1, num_iter + 1):
    ##############################################################
    out = model(inp)
    
    loss = criterion(out, Variable(torch.Tensor([float(pred)]).cuda().long()))
    loss.backward()
    
    # this is the method			
    perturbation = (alpha/255.0) * torch.sign(inp.grad.data)
    perturbation = torch.clamp((inp.data + perturbation) - orig.data, min=-eps/255.0, max=eps/255.0)
    inp.data = orig.data + perturbation
    
    inp.grad.data.zero_() 
    ################################################################
    
    # predict on the adversarial image, this inp is not the adversarial example we want, it's not yet clamped. And clamping can be done only after deprocessing.
    pred_adv = np.argmax(model(inp).data.cpu().numpy())
    
    print("Iter [%3d/%3d]:  Prediction: %s"%(i, num_iter, classes[pred_adv].split(',')[0]))
    
    # deprocess image
    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                         transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                         transforms.Lambda(clip)])
    adv = tf(inp.data.cpu()[0]).numpy().transpose(1, 2, 0)
    pert = (adv - img.data.cpu().numpy().transpose(1, 2, 0))
    adv = adv * 255.0
    adv = adv[..., ::-1] # RGB to BGR
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    if args.save and i == 20:
        print('==> write out images...')
        
        adv_folder = os.path.join(output_folder, 'adv-img-folder')
        final_adv_folder = os.path.join(adv_folder, 'images')
        
        cv2.imwrite(os.path.join(final_adv_folder, 
            os.path.splitext(file_name)[0] + '.' +
            short_name + '.ifgsm.e' + str(eps).zfill(3) + '.i' + str(i).zfill(4) + '.png'), adv)
        pert_folder = os.path.join(output_folder, 'per-img-folder')
        cv2.imwrite(os.path.join(pert_folder,
            os.path.splitext(file_name)[0] + '.' +
            short_name + '.ifgsm.e' + str(eps).zfill(3) + '.i' + str(i).zfill(4) + '.pert.png'), pert)
print()

