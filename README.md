# GPCR_GK-means_fast_GK-means
This repo is an implemenataion of Gaussian smoothing and PNM color reduction (GPCR), color quantization using Gaussian smoothing and K-means (GK-means), and fast GK-means in our paper: "Leveraging Image Processing Techniques to Thwart Adversarial Attacks in Image Classification" [1].

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
We adopt Iterative implementations of Fast Gradient Sign Method (IFGSM) and DeepFool from two Github implemetations [2][3] accordingly. Our implementations enable users to take one or more input images to craft adversarial images.

Note: Before running two shell scripts, users need to adapt a directory path accordingly.

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

Our three proposed approaches


**(3) Image Classification** <br/>

Note: put the reference.....





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
```
