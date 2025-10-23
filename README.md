# hybrid-curriculum-learning
This repository contains code and artifacts related to the course project for CMU's 11-785 Intro to Deep Learning course. In this project we are going to explore the performance of curriculum learning approaches (e.g., [data/blur based](https://proceedings.neurips.cc/paper/2020/file/f6a673f09493afcd8b129a0bcf1cd5bc-Paper.pdf) as well as a [Learning Rate Curriculum](https://arxiv.org/pdf/2205.09180)). In our main investigation we will seek to combine these two approaches and evaluate the impact on various models/architectures used for image classification (e.g., ResNet,CvT-13, ConvNeXt).

## Structure
- models (this contains training notebooks for our approach on various model architecture)
  - cvt-13
    - imagenet-100 (for training notebooks that use the [imagenet-100](https://www.kaggle.com/datasets/wilyzh/imagenet100) dataset)
    - cifar-10 (for training notebooks that use the [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset)
  - resnet-18
    - ...
  - convnext-tiny
    - ...

## References:
- Also listed sporadically in notebooks
- CvT-13
    - paper: https://arxiv.org/abs/2103.15808
    - implementation: https://github.com/microsoft/CvT
- Learning Rate Curriculum (LeRaC)
    - paper: https://arxiv.org/abs/2205.09180
    - implementation: https://github.com/CroitoruAlin/LeRaC
- Improving ResNet Model Accuracy with Curriculum Learning Using Blurred Images
    - paper: https://dl.acm.org/doi/10.1145/3606043.3606086
- Datasets:
    - CIFAR-10: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
    - ImageNet-100: https://www.kaggle.com/datasets/wilyzh/imagenet100
