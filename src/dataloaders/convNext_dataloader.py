from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch

SEED = 42


def get_imagenet100_loaders(config):
    """
    Load ImageNet-100 dataset with Curriculum (Variable K) Blur.
    
    Returns:
        train_loader: Dynamic loader. Call train_loader.dataset.set_epoch(epoch) each epoch.
        val_loader: Standard validation loader.
    """
    print("Loading ImageNet-100 Curriculum dataset...")

    data_path = config['dataset_path']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    # Configuration for Curriculum
    k_epochs = config.get('curriculum_k', 5)  # Duration of curriculum
    max_sigma = config.get('max_sigma', 2.0)  # Max blur strength
    
    # 1. Define Static Transforms
    # Split into Base (Geometric/Color) and Final (Tensor/Norm)
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Base Augmentations (applied BEFORE blur)
    base_aug_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # Final Transforms (applied AFTER blur)
    final_tfms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms (Standard)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 2. Define Curriculum Wrapper
    class CurriculumImageFolder(Dataset):
        def __init__(self, root, base_aug, final_tfms, k_epochs, max_sigma=2.0):
            # Load underlying ImageFolder without transforms to get raw PIL images
            self.dataset = datasets.ImageFolder(root, transform=None)
            self.base_aug = base_aug
            self.final_tfms = final_tfms
            
            # Curriculum state
            self.k = k_epochs
            self.max_sigma = max_sigma
            self.current_epoch = 0
            
            # Expose classes for convenience
            self.classes = self.dataset.classes
            
        def set_epoch(self, epoch):
            self.current_epoch = epoch
            
        def __getitem__(self, idx):
            img, target = self.dataset[idx] # Returns PIL Image
            
            # A. Base Augmentations
            img = self.base_aug(img)
            
            # B. Dynamic Curriculum Blur
            if self.current_epoch < self.k:
                progress = self.current_epoch / self.k
                current_sigma = self.max_sigma * (1.0 - progress)
                
                if current_sigma > 0.1:
                    # Kernel size 21 is more appropriate for 224x224 resolution 
                    # than 5 to see visible blur effects.
                    bluror = transforms.GaussianBlur(kernel_size=21, sigma=current_sigma)
                    img = bluror(img)
            
            # C. Finalize
            img = self.final_tfms(img)
            
            return img, target
            
        def __len__(self):
            return len(self.dataset)

    # 3. Instantiate Datasets
    train_dir = Path(data_path) / 'train'
    val_dir = Path(data_path) / 'val'
    
    # The Dynamic Training Set
    train_dataset = CurriculumImageFolder(
        root=train_dir,
        base_aug=base_aug_tfms,
        final_tfms=final_tfms,
        k_epochs=k_epochs,
        max_sigma=max_sigma
    )
    
    # Standard Validation Set
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
    
    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset, 
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

    # Printing Stats
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Curriculum Config: k={k_epochs} epochs, Max Sigma={max_sigma}")
    
    return train_loader, val_loader




def get_cifar_loaders(config):
    """
    Returns:
        train_loader: A dynamic loader. You MUST call train_loader.dataset.set_epoch(epoch) 
                      at the start of each training epoch.
        val_loader: Standard validation loader.
        test_loader: Standard test loader.
    """
    
    # 1. Define Static Transforms (Base Augmentations & Normalization)
    # We split these so we can inject the Blur in the middle.
    
    # Base Augmentations (Geometric/Color)
    base_aug_tfms = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    ])

    # Final Prep (Tensor conversion + Normalization)
    final_tfms = transforms.Compose([
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

    # 2. Define the Dynamic Curriculum Dataset
    class CurriculumDataset(Dataset):
        def __init__(self, base_dataset, indices, base_aug, final_tfms, k_epochs, max_sigma=2.0):
            self.base = base_dataset
            self.indices = indices
            self.base_aug = base_aug
            self.final_tfms = final_tfms
            
            # Curriculum State
            self.k = k_epochs
            self.max_sigma = max_sigma
            self.current_epoch = 0
            
        def set_epoch(self, epoch):
            """Call this at the start of every training epoch."""
            self.current_epoch = epoch

        def __getitem__(self, i):
            img, target = self.base[self.indices[i]]
            
            # Step A: Apply standard geometric augmentations (on PIL image)
            img = self.base_aug(img)
            
            # Step B: Apply Dynamic Gaussian Blur (The Curriculum)
            # Logic: Decay sigma linearly from max_sigma to 0 over k epochs
            if self.current_epoch < self.k:
                # Calculate progress (0.0 to 1.0)
                progress = self.current_epoch / self.k
                current_sigma = self.max_sigma * (1.0 - progress)
                
                # Apply blur if sigma is significant (min sigma for GaussianBlur is usually > 0)
                if current_sigma > 0.1:
                    # kernel_size must be odd. 5 is a standard choice for CIFAR size.
                    blur_layer = transforms.GaussianBlur(kernel_size=5, sigma=current_sigma)
                    img = blur_layer(img)
            
            # Step C: Finalize (Tensor + Normalize)
            img = self.final_tfms(img)
            
            return img, target

        def __len__(self):
            return len(self.indices)

    # 3. Setup Split
    root = config.get('dataset_path', './data')
    seed = config.get('seed', 42)
    
    # Base dataset (no transforms applied yet)
    base_train = datasets.CIFAR10(root, train=True, download=True, transform=None)

    val_len = int(len(base_train) * config["val_split_ratio"])
    train_len = len(base_train) - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(len(base_train)), [train_len, val_len], generator=generator
    )
    
    # 4. Instantiate Datasets
    # We define a helper for validation that is static
    class StaticDataset(Dataset):
        def __init__(self, base, indices, tfm):
            self.base = base
            self.indices = indices
            self.tfm = tfm
        def __getitem__(self, i):
            img, tgt = self.base[self.indices[i]]
            return self.tfm(img), tgt
        def __len__(self):
            return len(self.indices)

    # The dynamic training set
    train_ds = CurriculumDataset(
        base_dataset=base_train,
        indices=train_subset.indices,
        base_aug=base_aug_tfms,
        final_tfms=final_tfms,
        k_epochs=config.get("curriculum_k", 5), # Default to 5 epochs if not set
        max_sigma=config.get("max_sigma", 2.0)
    )

    val_ds = StaticDataset(base_train, val_subset.indices, val_tfms)
    test_ds = datasets.CIFAR10(root, train=False, download=True, transform=val_tfms)

    # 5. Build Loaders
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    
    return train_loader, val_loader, test_loader