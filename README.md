# Leveraging image processing techniques to thwart adversarial attacks in image classification (ISM 2019)
This repo is an implemenataion of Gaussian smoothing and PNM color reduction (GPCR), color quantization using Gaussian smoothing and K-means (GK-means), and fast GK-means in our paper: "**Leveraging Image Processing Techniques to Thwart Adversarial Attacks in Image Classification**" [1].

![Image description](/overall_results.png)

# Abstract
Deep Convolutional Neural Networks (DCNNs) are vulnerable to images that have been altered with well-engineered and imperceptible perturbations. We propose three color quan- tization pre-processing techniques to make DCNNs more robust to adversarial perturbation including Gaussian smoothing and PNM color reduction (GPCR), color quantization using Gaussian smoothing and K-means (GK-means), and fast GK-means. We evaluate the approaches on a subset of the ImageNet dataset. Our evaluation reveals that our GK-means-based algorithms have the best top-1 accuracy. We also present the trade-off between GK-means-based algorithms and GPCR with respect to computational time.

# Setup

**(1) Download validation images from Imagenet** <br/>
Please visit the website http://image-net.org/download-images and sign up an account to download all validation images (50,000 images).

**(2) Random Image Selection** <br/>
An algorithm to make a new dataset with few images:
1. run "choose_imgs.py" to choose the squarest images.
2. run "run_move_rename_imgs.sh" to rename all images.
3. run "img_classification.sh" to generate lists.
4. run "img_check_dataset.awk" to have common images.
5. run "choose_imgs.py" to choose a subset of images. (e.g. 1000)
6. run "move_imgs.sh" to create a smaller dataset from 50,000 images.

Specifically, please run the following commands
1. calculate the ratio of the short side to the long side for each image
```
./calculate_ratio.sh <a directory to input images>
```

2. choose the squarest images
```
python3 choose_imgs.py <a path to an input file> <a smaller number e.g. 5000> <a bigger number e.g. 5500>
```

3. rename all selected images from step 2
```
./run_move_rename_imgs.sh <a list with 1000 or 5000 selected images> <a path to an output directory>
```

4. perform an image classification task to generate a result list
```
./img_classification.sh <a directory path to input images> <deep learning model e.g. inception_v3> 
<a flag to load preprocessed images (yes or no)> <a file path to classification results>
```

5. choose a small number of images randomly
```
python3 choose_imgs.py <a path to an input file> <a smaller number e.g. 1000> <a bigger number e.g. 2500>
```

6. make the new dataset with few images from 50,000 images
```
./move_imgs.sh <a path to an input file with a list of images>
```

# Usage

**(1) Adversarial Image Generation** <br/>
We adopt Iterative Fast Gradient Sign Method (IFGSM) and DeepFool from two Github implemetations [2][3] accordingly. Our implementations enable users to take one or more input images to craft adversarial images. Before running two shell scripts, users need to adapt a directory path accordingly.

To perform the **IFGSM** attack, please run the following command
```
./adversarial_attack/run_ifgsm_attack.sh <a directory path to all input images> 
<a deep learning model e.g. resnet18 or inception_v3> <a directory path to adversarial images>
```

Similarly, plese run the following command to perform the **DeepFool** attack
```
./adversarial_attack/run_deepfool_attack.sh <a directory path to all input images> 
<a deep learning model e.g. resnet18 or inception_v3> <a directory path to adversarial images>
```

**(2) Remove Adversarial Perturbation** <br/>
We implemented our proposed approaches in python3 with OpenCV and shell. In order to use our implementations easily, we provided shell scripts to users. Please run the following commands to perform GPCR, GK-means, fast GK-means. For K-means, we followed the same idea in an OpenCV tutorial [4] to perform color quantization.

GPCR
```
./run_gpcr.sh <a directory path to input images> <the number of colors e.g. 64 or 128> 
<a directory path to output images>
```

GK-means
```
./run_gkmeans.sh <a directory path to input images> <the number of colors e.g. 64 or 128> 
<a directory path to output images>
```

Fast GK-means
```
./run_fast_gkmeans.sh <a directory path to input images> <the number of colors e.g. 64 or 128> 
<a directory path to output images>
```

**(3) Image Classification** <br/>
We adopt an implementation [5] for image classification and tranfer learning to be able to display top-5 classification results. Before running the folllowing command, users must refer to the implementation [5] to make two directories for train and val.

To perform image classification, please run the following command
```
./img_classification.sh <a directory path to input images> <deep learning model e.g. inception_v3> 
<a flag to load preprocessed images (yes or no)> <a file path to classification results>
```

To calculate top-1, top-3, and top-5 classification accuracy, please run the following command
```
awk -f top1_top5_calculation.awk <a path to classification result file>
```

# License
All the implementations are used for academic only. If you are interested in our work for other purposes or have any questions, please reach out the authors of the paper.

# Reference
```
[1]  @inproceedings{Jalalpour_ISM_2019,
         author = {Yeganeh Jalalpour and Li-Yun Wang and Ryan Feng and Wu-chi Feng},
         title = {Leveraging Image Processing Techniques to Thwart Adversarial Attacks in Image Classification},
         booktitle = {IEEE International Symposium on Multimedia},
         year = {2019}
     }
[2]  IFGSM Implementation: https://github.com/sarathknv/adversarial-examples-pytorch/tree/master/iterative
[3]  DeepFool Implementation: https://github.com/LTS4/DeepFool/tree/master/Python
[4]  OpenCV Tutorial https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
[5]  Image Classification for Imagenet https://github.com/floydhub/imagenet
```
