import torchvision
from torchvision import datasets, transforms as T
import logging
import torch

from timm.data import create_loader, create_transform
import os

class LinearlyDecayingGaussianBlur:
    """Linearly decreases blur application from 100% to 0% over blur_epochs."""
    def __init__(self, kernel_size, sigma, blur_epochs):
        self.sigma = sigma
        self.blur_epochs = blur_epochs
        self.current_epoch = 0
        self.kernel_size = kernel_size
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __call__(self, img):
        if self.current_epoch >= self.blur_epochs:
            return img
        blur_proportion = 1 - (self.current_epoch / self.blur_epochs)
        if torch.rand(1).item() < blur_proportion:
            return self.blur(img)
        return img

def build_gaussian_transforms_cifar(config):
    normalize = T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    if config['TRAIN']['BLUR']['USE_LINEAR_DECAY']:
        logging.info('Using linearly decaying Gaussian blur')
        linear_blur = LinearlyDecayingGaussianBlur(
            kernel_size=config['TRAIN']['BLUR']['KERNEL_SIZE'], 
            sigma=config['TRAIN']['BLUR']['SIGMA'],
            blur_epochs=config['TRAIN']['BLUR']['EPOCHS']
        )
        transforms = T.Compose([
            T.ToTensor(),
            linear_blur,
            normalize
        ])
        return transforms, linear_blur
    else:
        transforms = T.Compose([
            T.ToTensor(),
            T.GaussianBlur(kernel_size=config['TRAIN']['BLUR']['KERNEL_SIZE'], sigma=config['TRAIN']['BLUR']['SIGMA']),
            normalize
        ])
        return transforms, None

def build_gaussian_dataset_cifar(config, is_train=True):
    '''
    In the CIFAR file it will call the appropriate method
    '''
    dataset = None
    transforms, linear_blur = build_gaussian_transforms_cifar(config)
    dataset = datasets.CIFAR10(root=config['DATASET']['ROOT'], train=is_train, download=True, transform=transforms)
    logging.info(f'load samples: {len(dataset)}, is_train: {is_train}')
    return dataset, linear_blur    

def build_gaussian_dataloader_cifar(config):
    dataset, linear_blur = build_gaussian_dataset_cifar(config)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['TRAIN']['BATCH_SIZE_PER_GPU'],
        shuffle=True,
        num_workers=config['WORKERS'],
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )
    print(f"\nDataLoader Info:")
    print(f"Gaussian Loader batches: {len(data_loader)}")
    print(f"Gaussian dataset size: {len(data_loader.dataset)}")
    print(f"Gaussian Dataset of classes: {len(data_loader.dataset.classes)}")
    data_loader.linear_blur = linear_blur # will be None if not using linear decay
    # Test loading a batch
    images, labels = next(iter(data_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")        
    return data_loader    


def build_transforms_cifar(config, is_train):
    if is_train:
        img_size = config['TRAIN']['IMAGE_SIZE'][0]
        timm_cfg = config['AUG']['TIMM_AUG']
        # hardcoded values are from defaults e.g., https://github.com/microsoft/CvT/blob/main/lib/config/default.py#L68
        transforms = create_transform(
            input_size = img_size,
            is_training = True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg['RE_PROB'],
            re_mode=timm_cfg['RE_MODE'],
            re_count=timm_cfg['RE_COUNT'],
            scale=(0.8, 1.0),
            ratio=(3.0/4.0, 4.0/3.0),
            hflip=timm_cfg['HFLIP'],
            vflip=timm_cfg['VFLIP'],
            color_jitter=timm_cfg['COLOR_JITTER'],
            auto_augment=timm_cfg['AUTO_AUGMENT'],
            interpolation=timm_cfg['INTERPOLATION'],
            mean=(0.491, 0.482, 0.446), # https://stackoverflow.com/a/69699979
            std=(0.247, 0.243, 0.261),            
            
        )
    else:
        normalize = T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        img_size = config['TEST']['IMAGE_SIZE'][0]
        transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
    return transforms

def build_dataset_cifar(config, is_train):
    '''
    In the CIFAR file it will call the appropriate method
    '''
    dataset = None
    transforms = build_transforms_cifar(config, is_train)
    dataset = datasets.CIFAR10(root=config['DATASET']['ROOT'], train=is_train, download=True, transform=transforms)
    logging.info(f'load samples: {len(dataset)}, is_train: {is_train}')
    return dataset

def build_dataloader_cifar(config, is_train):
    if is_train:
        batch_size_per_gpu = config['TRAIN']['BATCH_SIZE_PER_GPU']
        shuffle = True
    else:
        batch_size_per_gpu = config['TEST']['BATCH_SIZE_PER_GPU']
        shuffle = False
    dataset = build_dataset_cifar(config, is_train)
    sampler = None # the cvt code has code to use multi-gpu. here we assume single A100 gpu
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=config['WORKERS'],
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )     
    return data_loader    

def get_cifar_dataloaders(config):
    train_loader = build_dataloader_cifar(config, is_train=True)
    val_loader = build_dataloader_cifar(config, is_train=False)    
    print(f"\nDataLoader Info:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")    
    return train_loader, val_loader


def build_gaussian_transforms_imagenet(config):
    normalize = T.Normalize(mean=[0.485,  0.456, 0.406], std=[0.229, 0.224, 0.225])
    if config['TRAIN']['BLUR']['USE_LINEAR_DECAY']:
        logging.info('Using linearly decaying Gaussian blur')
        linear_blur = LinearlyDecayingGaussianBlur(
            kernel_size=config['TRAIN']['BLUR']['KERNEL_SIZE'], 
            sigma=config['TRAIN']['BLUR']['SIGMA'],
            blur_epochs=config['TRAIN']['BLUR']['EPOCHS']
        )
        transforms = T.Compose([
            T.ToTensor(),
            linear_blur,
            normalize
        ])
        return transforms, linear_blur
    else:
        transforms = T.Compose([
            T.ToTensor(),
            T.GaussianBlur(kernel_size=config['TRAIN']['BLUR']['KERNEL_SIZE'], sigma=config['TRAIN']['BLUR']['SIGMA']),
            normalize
        ])
        return transforms, None

def build_gaussian_dataset_imagenet(config, is_train=True):
    transforms, linear_blur = build_gaussian_transforms_imagenet(config)
    dataset_name = config['DATASET']['TRAIN_SET'] # will only use blur dataset for training
    dataset = datasets.ImageFolder(os.path.join(config['DATASET']['ROOT'], dataset_name), transforms)
    logging.info(f'load samples: {len(dataset)}, is_train: {is_train}')
    return dataset, linear_blur    

def build_gaussian_dataloader_imagenet(config):
    dataset, linear_blur = build_gaussian_dataset_imagenet(config)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['TRAIN']['BATCH_SIZE_PER_GPU'],
        shuffle=True,
        num_workers=config['WORKERS'],
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )
    data_loader.linear_blur = linear_blur # will be None if not using linear decay
    print(f"\nDataLoader Info:")
    print(f"Gaussian Loader batches: {len(data_loader)}")
    print(f"Gaussian dataset size: {len(data_loader.dataset)}")
    print(f"Gaussian Dataset of classes: {len(data_loader.dataset.classes)}")

    # Test loading a batch
    images, labels = next(iter(data_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")          
    return data_loader
    
  
def build_transforms_imagenet(config, is_train):
    if is_train:
        img_size = config['TRAIN']['IMAGE_SIZE'][0]
        timm_cfg = config['AUG']['TIMM_AUG']
        # hardcoded values are from defaults e.g., https://github.com/microsoft/CvT/blob/main/lib/config/default.py#L68
        transforms = create_transform(
            input_size = img_size,
            is_training = True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg['RE_PROB'],
            re_mode=timm_cfg['RE_MODE'],
            re_count=timm_cfg['RE_COUNT'],
            scale=(0.08, 1.0),
            ratio=(3.0/4.0, 4.0/3.0),
            hflip=timm_cfg['HFLIP'],
            vflip=timm_cfg['VFLIP'],
            color_jitter=timm_cfg['COLOR_JITTER'],
            auto_augment=timm_cfg['AUTO_AUGMENT'],
            interpolation=timm_cfg['INTERPOLATION'],
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),            
            
        )
    else:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_size = config['TEST']['IMAGE_SIZE'][0]
        transforms = T.Compose([
            T.Resize(int(img_size/ 0.875), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize
        ])
    return transforms

def _build_imagenet_dataset(config, is_train):
    transforms = build_transforms_imagenet(config, is_train)
    dataset_name = config['DATASET']['TRAIN_SET'] if is_train else config['DATASET']['TEST_SET']
    dataset = datasets.ImageFolder(os.path.join(config['DATASET']['ROOT'], dataset_name), transforms)
    logging.info(
        'load samples: {}, is_train: {}'
        .format(len(dataset), is_train)
    )
    
    return dataset
    
def build_dataset_imagenet(config, is_train):
    '''
    In this file this will call the build_imagenet_dataset method.
    In the CIFAR file it will call the appropriate method
    '''
    dataset = None
    dataset = _build_imagenet_dataset(config, is_train)
    return dataset

def build_dataloader_imagenet(config, is_train):
    if is_train:
        batch_size_per_gpu = config['TRAIN']['BATCH_SIZE_PER_GPU']
        shuffle = True
    else:
        batch_size_per_gpu = config['TEST']['BATCH_SIZE_PER_GPU']
        shuffle = False
    dataset = build_dataset_imagenet(config, is_train)
    sampler = None # the cvt code has code to use multi-gpu. here we assume single A100 gpu
    # set to true in the config so we are going to use TIMM loader for training
    if is_train:
        logging.info('use timm loader for training')
        timm_cfg = config['AUG']['TIMM_AUG']
        data_loader = create_loader(
            dataset,
            input_size=tuple(config['TRAIN']['IMAGE_SIZE']),
            batch_size=config['TRAIN']['BATCH_SIZE_PER_GPU'],
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg['RE_PROB'],
            re_mode=timm_cfg['RE_MODE'],
            re_count=timm_cfg['RE_COUNT'],
            re_split=timm_cfg['RE_SPLIT'],
            scale=(0.08, 1.0),
            ratio=(3.0/4.0, 4.0/3.0),
            hflip=timm_cfg['HFLIP'],
            vflip=timm_cfg['VFLIP'],
            color_jitter=timm_cfg['COLOR_JITTER'],
            auto_augment=timm_cfg['AUTO_AUGMENT'],
            interpolation=timm_cfg['INTERPOLATION'],
            num_aug_splits=0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            num_workers=config['WORKERS'],
            distributed=False,
            collate_fn=None,
            pin_memory=True,
            use_multi_epochs_loader=True
        )        
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=config['WORKERS'],
            pin_memory=True,
            sampler=sampler,
            drop_last=True if is_train else False,
        )     
    return data_loader

def get_imagenet_dataloaders(config):
    train_loader = build_dataloader_imagenet(config, is_train=True)
    val_loader = build_dataloader_imagenet(config, is_train=False)

    print(f"\nDataLoader Info:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")    
    return train_loader, val_loader
