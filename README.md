# GPCR_GK-means_fast_GK-means
This repo is an implemenataion of Gaussian smoothing and PNM color reduction (GPCR), color quantization using Gaussian smoothing and K-means (GK-means), and fast GK-means in our paper: "Leveraging Image Processing Techniques to Thwart Adversarial Attacks in Image Classification" [1].

![Image description](/overall_results.png)

# Abstract
Deep Convolutional Neural Networks (DCNNs) are vulnerable to images that have been altered with well-engineered and imperceptible perturbations. We propose three color quan- tization pre-processing techniques to make DCNNs more robust to adversarial perturbation including Gaussian smoothing and PNM color reduction (GPCR), color quantization using Gaussian smoothing and K-means (GK-means), and fast GK-means. We evaluate the approaches on a subset of the ImageNet dataset. Our evaluation reveals that our GK-means-based algorithms have the best top-1 accuracy. We also present the trade-off between GK-means-based algorithms and GPCR with respect to computational time.

# Setup

**(1) Download validation images from Imagenet** <br/>
Please visit the website http://image-net.org/download-images and sign up an account to download all validation images (50,000 images).

**(2) Random Image Selection**




# Usage

**(1) Adversarial Image Generation**
We adopt Iterative implementations of Fast Gradient Sign Method (IFGSM) and DeepFool from two Github implemetations [2][3] accordingly. Our implementations enable users to take one or more input images to craft adversarial images.

To perform the **IFGSM** attack, please run the following command
```
```

Similarly, plese run the following command to perform the **DeepFool** attach
```
```

**(2) Remove Adversarial Perturbation**

Our three proposed approaches


**(3) Image Classification**
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
[2]  IFGSM https://github.com/sarathknv/adversarial-examples-pytorch/tree/master/iterative
[3]  DeepFool https://github.com/LTS4/DeepFool/tree/master/Python
```
