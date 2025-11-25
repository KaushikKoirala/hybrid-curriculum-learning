import torch
from torchsummary import summary
import torchvision
from torchvision.utils import make_grid
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import gc
# from tqdm import tqdm
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
import wandb
import matplotlib.pyplot as plt
from pytorch_metric_learning import samplers
import csv
import logging
from timm.data import create_loader, create_transform
import torch.nn as nn
from functools import partial
import torch
from einops import rearrange
from einops.layers.torch import Rearrange

from config import cvt_13_cifar_config, cvt_13_imagenet_config
from dataloaders import dataloader
from models import cvt_13, params
from optimizers import cvt_optimizer
from schedulers import cvt_schedule
import argparse 
from criterion import build_criterion
from timm.data.mixup import Mixup
from metrics import AverageMeter, accuracy
from model_utils import save_model, load_model
import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)
def train_one_epoch(epoch, model, train_loader, gaussian_loader, criterion, optimizer, scheduler, 
                    config, scaler, mixup_fn):
    """
    Train for one epoch
    Following: https://github.com/microsoft/CvT/blob/main/lib/core/function.py
    """

    losses = AverageMeter()
    acc_m = AverageMeter()

    model.train()
    accumulation_steps = config['TRAIN'].get('GRADIENT_ACCUMULATION_STEPS', 1)
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
    loader = gaussian_loader if epoch < config['TRAIN']['BLUR']['EPOCHS'] else train_loader
    for idx, (images, targets) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        images, targets = mixup_fn(images, targets)

        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
   
            loss = criterion(outputs, targets)
       
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if idx % 100 == 0 and epoch >= 6:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Epoch {epoch}, Batch {idx}, Grad norm: {total_norm:.4f}")            
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-0.5, 0.5)                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()        

        targets_for_acc = torch.argmax(targets, dim=1)

        acc = accuracy(outputs, targets_for_acc)

        losses.update(loss.item()*accumulation_steps, images.size(0))
        acc_m.update(acc[0].item(), images.size(0))
        batch_bar.set_postfix(
            acc="{:.02f}% ({:.02f}%)".format(acc[0].item(), acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item()*accumulation_steps, losses.avg),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar        
    
    if (idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-0.5, 0.5)                    
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)                    
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    batch_bar.close()

    scheduler.step()
    torch.cuda.empty_cache()

    return losses.avg, acc_m.avg

@torch.no_grad()
def validate(model, val_loader, criterion, config):
    losses = AverageMeter()
    acc_m = AverageMeter()
    
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val.', ncols=5)

    for idx, (images, targets) in enumerate(val_loader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        acc = accuracy(outputs, targets)
        losses.update(loss.item(), images.size(0))
        acc_m.update(acc[0].item(), images.size(0))
        batch_bar.set_postfix(
            acc="{:.02f}% ({:.02f}%)".format(acc[0].item(), acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), losses.avg))

        batch_bar.update()

    batch_bar.close()
    print(f' * Acc {acc_m.avg:.3f}')
    
    return losses.avg, acc_m.avg


def do_training_loop(model, cfg, train_loader, gaussian_loader, val_loader, criterion, criterion_eval, optimizer, scheduler, mixup_fn, run_id):
    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)
    model = model.to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()    
    wandb.login(key=os.environ.get('WANDB_API_KEY')) # API Key is in your wandb account, under settings (wandb.ai/settings)
    run = wandb.init(
        name = f"idl-project-cvt-13-{run_id}", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
    #     id = "t0xoatlu", #Insert specific run id here if you want to resume a previous run
    #     resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "idl-project", ### Project should be created in your wandb account
        config = cfg ### Wandb Config for your run
    )
    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_loss = -1
    start_epoch = cfg['TRAIN']['BEGIN_EPOCH']
    end_epoch = cfg['TRAIN']['END_EPOCH']

    print("Starting Training")
    print(f"Epochs: {start_epoch} -> {end_epoch}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Device: {DEVICE}")

    for epoch in range(start_epoch, end_epoch):
        # epoch
        print("\nEpoch {}/{}".format(epoch+1, end_epoch))
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, gaussian_loader, criterion, optimizer, scheduler, 
                        cfg, scaler, mixup_fn)
        
        val_loss, val_acc = validate(model, val_loader, criterion_eval, cfg)
        is_best = (best_loss) == -1 or val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if epoch %20 == 0:
            save_model(model, optimizer, scheduler, epoch,os.path.join(cfg['OUTPUT_DIR'], f'{epoch}.pth'))
        if is_best:
            save_model(model, optimizer, scheduler, epoch, os.path.join(cfg['OUTPUT_DIR'], 'best.pth') )

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
        metrics = {'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'epoch': epoch}
        if run is not None:
            run.log(metrics)

@torch.no_grad()
def test(model, test_loader, config):

    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for images, targets in tqdm(test_loader, desc='Testing'):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        probs = torch.softmax(outputs, dim=1)
        
        _, preds = outputs.topk(1, 1, True, True)
        
        all_predictions.extend(preds.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())
        
        acc1 = accuracy(outputs, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
    
    return {
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities),
        'loss': losses.avg,
        'top1_acc': top1.avg
    }



def main():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument('-d', '--dataset', type=str, default='cifar')
    parser.add_argument('-n', '--run_name', type=str)
    parser.add_argument('-l', '--lerac_epochs', type=int, default=5)
    parser.add_argument('-b', '--blur_epochs', type=int, default=20)
    parser.add_argument('-c', '--c_factor', type=float, default=10.0)
    parser.add_argument('-e', '--eta_0', type=float, default=.002)
    args = parser.parse_args()
    run_logical_id = f'{args.dataset}_hybrid_{args.run_name}'
    if args.dataset.lower() == 'cifar':
        cfg = cvt_13_cifar_config.get_cvt_13_cifar_config(run_id=run_logical_id, lerac_epochs=args.lerac_epochs, blur_epochs=args.blur_epochs)
        gaussian_dataloader = dataloader.build_gaussian_dataloader_cifar(cfg)
        train_loader, val_loader = dataloader.get_cifar_dataloaders(cfg)
        num_classes = 10
    elif args.dataset.lower() == 'imagenet':
        cfg = cvt_13_imagenet_config.get_cvt_13_imagenet_config(run_id=run_logical_id, lerac_epochs=args.lerac_epochs, blur_epochs=args.blur_epochs)
        gaussian_dataloader = dataloader.build_gaussian_dataloader_imagenet(cfg)
        train_loader, val_loader = dataloader.get_imagenet_dataloaders(cfg)
        num_classes = 100
    cvt_13_model = cvt_13.create_cvt_13(cfg, num_classes=num_classes)
    params.count_parameters(cvt_13_model)
    optimizer = cvt_optimizer.get_cvt_optimizer(cfg, cvt_13_model, eta_0=args.eta_0)
    scheduler = cvt_schedule.get_cvt_scheduler(cfg, optimizer, c_factor=args.c_factor)
    criterion = build_criterion()
    criterion.cuda()
    criterion_eval = build_criterion(is_train=False)
    criterion_eval.cuda()
    aug = cfg['AUG']
    mixup_fn = Mixup(
        mixup_alpha=aug['MIXUP'], cutmix_alpha=aug['MIXCUT'],
        cutmix_minmax=None,
        prob=aug['MIXUP_PROB'],
        label_smoothing=0.0,
        num_classes=num_classes
    )
    wandb_config = copy.deepcopy(cfg)
    wandb_config['lerac_epochs'] = args.lerac_epochs
    wandb_config['blur_epochs'] = args.blur_epochs
    wandb_config['c_factor'] = args.c_factor
    wandb_config['eta_0'] = args.eta_0
    do_training_loop(cvt_13_model, wandb_config, train_loader, gaussian_dataloader, val_loader, criterion, criterion_eval, optimizer, scheduler, mixup_fn, run_id=run_logical_id)    
    model, optimizer, scheduler, epoch = load_model(cvt_13_model, optimizer, scheduler, path=f"./OUTPUT/{run_logical_id}/best.pth")
    test_results = test(model, val_loader, cfg)

    print("Final Test Results")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Acc@1: {test_results['top1_acc']:.3f}%")

if __name__ == '__main__':
    main()
