import math, os, time, copy, random
import gc
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import random_split, DataLoader
from tqdm.auto import tqdm
from contextlib import nullcontext
import matplotlib.pyplot as plt

import argparse
from config.convNext_cifar_config import get_convNext_cifar_config
from config.convNext_imagenet_config import get_convNext_imagenet_config
from dataloaders.convNext_dataloader import get_cifar_loaders, get_imagenet100_loaders
from models.convNext_tiny import create_convNextT
from optimizers.convNext_optimizer import get_convNextT_optimizer
from schedulers.convNext_scheduler import get_convNextT_scheduler
from model_utils import save_model, load_model
from snr_utils import calculate_layer_snr, visualize_snr

SEED = 42
random.seed(SEED);  torch.manual_seed(SEED);  torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def accuracy(preds, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
        pred   = pred.t()
        correct= pred.eq(targets.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().mean().item()*100. for k in topk]


def run_epoch(config, loader, model, scaler, optimizer, scheduler, criterion, epoch:int=0, phase:str="train"):
    """
    If `optimizer` is given → training mode, otherwise evaluation mode.
    Memory-safe: no graph is kept when we don't need gradients.
    """

    train = optimizer is not None
    model.train(train)

    running_loss, running_acc = 0.0, 0.0
    steps = len(loader)

    bar = tqdm(loader, desc=f"{phase.title():>5} | Epoch {epoch:02}", leave=False)

    # Choose the right context managers
    grad_ctx = nullcontext() if train else torch.no_grad()
    amp_ctx  = torch.amp.autocast(device_type="cuda",
                                  dtype=torch.float16,
                                  enabled=config["amp"] and torch.cuda.is_available())

    with grad_ctx:
        for images, labels in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            with amp_ctx:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item()
            running_acc  += accuracy(outputs, labels)[0]
            bar.set_postfix(loss=f"{loss.item():.4f}")

    torch.cuda.empty_cache()     # free any leftover cached blocks
    return running_loss/steps, running_acc/steps

def run_snr_analysis(config, device):
    print("\n--- Starting SNR Analysis ---")
    
    # 1. Load Data
    # Use validation loader for stable statistics
    if config['dataset_path'] == "ImageNet100_224":
        _, val_loader = get_imagenet100_loaders(config)
    else:
        _, val_loader, _ = get_cifar_loaders(config)

    # 2. Initialize Models
    # Model A: Randomly Initialized (The "Noisy" state at Epoch 0)
    model_random = create_convNextT(config).to(device)
    
    # Model B: Converged / Best Model (The "Clean Signal" reference)
    model_best = create_convNextT(config).to(device)
    
    # Load weights for Model B
    ckpt_path = f"{config['ckpt_dir']}/best_convnext_tiny.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Handle cases where checkpoint saves state_dict inside a key
        if 'model_state_dict' in checkpoint:
            model_best.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_best.load_state_dict(checkpoint)
        print("Loaded best checkpoint for SNR reference.")
    else:
        print("Warning: Best checkpoint not found. Using random weights as reference (Analysis will be meaningless).")

    # 3. Calculate SNR
    # This compares how much "noise" the random initialization adds relative to the trained features
    layers, snrs = calculate_layer_snr(model_random, model_best, val_loader, device)

    # 4. Plot
    save_loc = f"{config['ckpt_dir']}/snr_decay_plot.png"
    visualize_snr(layers, snrs, save_path=save_loc)
                  

def main():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument('-d', '--dataset', type=str, default='cifar')
    parser.add_argument('-n', '--run_name', type=str)
    parser.add_argument('-l', '--lerac_epochs', type=int, default=5)
    parser.add_argument('-b', '--blur_epochs', type=int, default=20)
    parser.add_argument('-c', '--c_factor', type=float, default=10.0)
    parser.add_argument('-e', '--eta_min', type=float, default=2e-8)
    args = parser.parse_args()
    run_logical_id = f'{args.dataset}_hybrid_{args.run_name}'

    if args.dataset.lower() == 'cifar':
        config = get_convNext_cifar_config(run_id=run_logical_id, lerac_epochs=args.lerac_epochs, blur_epochs=args.blur_epochs, eta_min=args.eta_min)
        train_loader, val_loader, test_loader = get_cifar_loaders(config)
    elif args.dataset.lower() == 'imagenet':
        config = get_convNext_imagenet_config(run_id=run_logical_id, lerac_epochs=args.lerac_epochs, blur_epochs=args.blur_epochs, eta_min=args.eta_min)
        train_loader, val_loader = get_imagenet100_loaders(config)

    checkpoint_dir = config['ckpt_dir']

    model = create_convNextT(config)
    model = model.to(device)

    optimizer = get_convNextT_optimizer(config, model)
    scheduler =get_convNextT_scheduler(config, optimizer, len(train_loader))

    criterion  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    # TODO: implement mixup here

    print(checkpoint_dir)
    print(os.path.exists(checkpoint_dir))

    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    best_val_acc = 0.0
    patience = 16
    epoches_no_improve = 0

    history = {"train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": []}

    for epoch in range(1, config["num_epochs"]+1):
        t0 = time.time()

        train_loader.dataset.set_epoch(epoch - 1)

        tr_loss, tr_acc = run_epoch(config, train_loader, model, scaler, optimizer, scheduler, criterion, epoch, "train")
        val_loss, val_acc= run_epoch(config, val_loader, model, scaler, optimizer, scheduler, criterion, epoch, "val")

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss);   history["val_acc"].append(val_acc)
        
        metrics = {
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_loss,
        }

        if val_acc >= best_val_acc:
            epoches_no_improve = 0
            best_val_acc = val_acc

            save_model(model, optimizer, scheduler, metrics, epoch, f"{config['ckpt_dir']}/best_convnext_tiny.pth")
            print("Saved best val acc model")

        else:
            epoches_no_improve += 1

        save_model(model, optimizer, scheduler, metrics, epoch, f"{config['ckpt_dir']}/current_epoch.pth")
        print(f"Saved epoch {epoch} model")

        print(f"Epoch {epoch:02}/{config['num_epochs']} "
            f"| train loss {tr_loss:.4f} acc {tr_acc:.2f}% "
            f"| val loss {val_loss:.4f} acc {val_acc:.2f}% "
            f"| lr {scheduler.get_last_lr()[0]:.2e} "
            f"| time {(time.time()-t0):.1f}s")

        if epoches_no_improve >= patience:
            print("Early stopping")
            break

    # Plot the training and validation accuracy
    print("Plotting training history...")
    plt.figure(figsize=(10, 6))
    
    # Generate x-axis (epochs)
    epochs_range = range(1, len(history["train_acc"]) + 1)
    
    plt.plot(epochs_range, history["train_acc"], label='Training Accuracy', color='blue')
    plt.plot(epochs_range, history["val_acc"], label='Validation Accuracy', color='orange')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plot_path = f"{config['ckpt_dir']}/accuracy_plot.png"
    plt.savefig(plot_path)
    print(f"Accuracy plot saved to {plot_path}")
    plt.close()


    if args.dataset.lower() == 'cifar':
        test_loss, test_acc = run_epoch(test_loader, model, None)
        print(f"Test  - loss: {test_loss:.4f} - accuracy: {test_acc:.2f}%")
    
    run_snr_analysis(config, device)

if __name__ == '__main__':
    main()
