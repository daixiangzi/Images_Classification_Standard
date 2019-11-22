# pytorch-classification
Classification on CIFAR-10/100 and ImageNet with PyTorch.
copy from  https://github.com/bearpaw/pytorch-classification.git

## Features
* Unified interface for different network architectures
* Multi-GPU support
* Training progress bar with rich info
* Training log and training curve visualization code (see `./utils/logger.py`)  

## Training  
python3 cifar.py

## Results

### CIFAR
Top1 error rate on the CIFAR-10/100 benchmarks are reported. You may get different results when training your models with different random seed.
Note that the number of parameters are computed on the CIFAR-10 dataset.

| Model                     | Params (M)         |  CIFAR-10 (%)      | CIFAR-100 (%)      |
| -------------------       | ------------------ | ------------------ | ------------------ |
| alexnet                   | 2.47               | 22.78              | 56.13              |
| vgg19_bn                  | 20.04              | 6.66               | 28.05              |
| ResNet-110                | 1.70               | 6.11               | 28.86              |
| PreResNet-110             | 1.70               | 4.94               | 23.65              |
| WRN-28-10 (drop 0.3)      | 36.48              | 3.79               | 18.14              |
| ResNeXt-29, 8x64          | 34.43              | 3.69               | 17.38              |
| ResNeXt-29, 16x64         | 68.16              | 3.53               | 17.30              |
| DenseNet-BC (L=100, k=12) | 0.77               | 4.54               | 22.88              |
| DenseNet-BC (L=190, k=40) | 25.62              | 3.32               | 17.17              |


### ImageNet
Single-crop (224x224) validation error rate is reported.
| Model                | Params (M)         |  Top-1 Error (%)   | Top-5 Error  (%)   |
| -------------------  | ------------------ | ------------------ | ------------------ |
| ResNet-18            | 11.69              |  30.09             | 10.78              |
| ResNeXt-50 (32x4d)   | 25.03              |  22.6              | 6.29               |

### Support  
activation:Relu,Selu,Swish,Mish  
optimizer:SGD,Adam,Radam,adamW(+warm_up)  
init:Kaiming  
other:cutout  

