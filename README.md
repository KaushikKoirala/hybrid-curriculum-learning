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

## Running Training  
In the [src](./src) directory found at the root of this repository there are `_runner` python scripts that act as
entrypoints for training the various models using the hybrid curriculum learning approach.

```
src
  - cvt_runner.py - Entrypoint for CVT Training
  - resnet18_runner.py - Entrypoint for ResNet18 training
  - convNext_runner.py - Entrypoint for ConvNeXt-Tiny training 
```

### Prerequisites
Running this training code relies on primarily 3 things. 

1. An environment in which all the modules/libraries/dependencies defined in [requirements.txt](./requirements.txt) are installed.

```bash
pip install -r ./requirements.txt
```

2. Installation of the datasets mentioned in [References](./README.md#references) to the location defined in the config files (e.g., in [cvt_13_imagenet_config.py](./src/config/cvt_13_imagenet_config.py))
```JSON
    "DATASET": {
        "DATASET": "imagenet",
        "DATA_FORMAT": "jpg",
        "ROOT": "./ImageNet100_224", # ensure ImageNet data is downloaded, resized and extracted to this path.
        "TEST_SET": "val",
        "TRAIN_SET": "train"
    },
``` 

> NOTE: Resizes to the appropriate size may need to be done using the resize script in [utils/resize_script.py](./utils/resize_script.py)

3. Lastly, the training scripts rely on a `WANDB_API_KEY` environment variable so that data about the training runs can be tracked in [WANDB](https://wandb.ai/home)
### Parameters and Example Invocations
There are 6 parameters for the hybrid curriculum learning training. They are:
- `--dataset` - The dataset to train the model with. Acceptable values are `cifar` and `imagenet`
- `--run_name` - The run name to label a particular training run with. This will impact the output directory names for parameter files and the logical name used in wandb for the run.
- `--lerac_epochs` - The number of epochs at the beginning of training that the Learning Rate Curriculum should apply for before equalizing the learning rate and letting the normal learning rate continue. 

- `--blur_epochs` - The number of epochs at the beginning of training that the Blur Curriculum should apply for.

- `--c_factor` - The multiplicative factor between epochs in the early learning rate curriculum

- `--eta_min` - The minimum learning rate that training decays to within layers in a network during the Learning Rate Curriculum epochs

- `--linear_blur_extension` - This parameter only exists in the `extension` [branch](https://github.com/KaushikKoirala/hybrid-curriculum-learning/tree/extension) and enables linearly decaying the proportion of blurred images in the dataset as the epochs progress. Currently, it is only supported for `CvT-13`. 

Example Invocation running CvT-13 training with all of the parameters above set
```bash
python3 src/cvt_runner.py \
    --dataset imagenet \
    -n run-id-8 \
    --lerac_epochs 5 \
    --blur_epochs 20 \
    --c_factor 10.0 \
    --eta_min "1e-8"
```
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
