from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import torch

SEED = 42

def get_imagenet100_loaders(config):
    """Load ImageNet-100 dataset with standard augmentation"""

    print("Loading ImageNet-100 dataset...")

    data_path = config['dataset_path']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    blur_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # adjust strength as desired
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dir = Path(data_path) / 'train'
    val_dir = Path(data_path) / 'val'
    
    train_dataset_normal = datasets.ImageFolder(train_dir, transform=transform_train)
    train_dataset_blur = datasets.ImageFolder(train_dir, transform=blur_transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    
    # Create data loaders
    train_loader_normal = DataLoader(
        train_dataset_normal, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_blur = DataLoader(
        train_dataset_blur, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset_normal)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset_normal.classes)}")
    print(f"Classes: {train_dataset_normal.classes[:10]}..." if len(train_dataset_normal.classes) > 10 else f"Classes: {train_dataset_normal.classes}")
    print(f"Training batches: {len(train_loader_normal)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader_normal, train_loader_blur, val_loader




def get_cifar_loaders(config):
    # Transforms -----------------------------------------------------------
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(config["image_size"] + 32),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616)),
    ])

    blur_train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # adjust strength as desired
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616)),
    ])
    from torch.utils.data import Dataset, Subset

    root = config['dataset_path']

    # Base dataset without transform so we can apply different transforms per split
    base_train = datasets.CIFAR10(root, train=True, download=True, transform=None)

    val_len   = int(len(base_train) * config["val_split_ratio"])
    train_len = len(base_train) - val_len

    # Create split indices (reproducible)
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(range(len(base_train)), [train_len, val_len],
                                            generator=generator)
    train_indices = train_subset.indices
    val_indices   = val_subset.indices

    class transformDataset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            img, target = self.base[self.indices[i]]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    # Build train (normal), train (blurred), and validation datasets
    train_ds_normal = transformDataset(base_train, train_indices, transform=train_tfms)
    train_ds_blur   = transformDataset(base_train, train_indices, transform=blur_train_tfms)
    val_ds          = transformDataset(base_train, val_indices,   transform=val_tfms)

    # Test set (uses val_tfms)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=val_tfms)

    # Dataloaders ----------------------------------------------------------
    train_loader_normal = DataLoader(
        train_ds_normal, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True)

    train_loader_blur = DataLoader(
        train_ds_blur, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True)

    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True)
    
    return train_loader_normal, train_loader_blur, val_loader, test_loader