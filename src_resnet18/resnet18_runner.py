import torch
import os
import gc
from tqdm.auto import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt

import sys
import os

# Add current directory (src_resnet18) to path - everything is self-contained
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from src_resnet18 modules (all self-contained)
from config.resnet18_cifar_config import get_resnet18_cifar_config
from models import resnet18, params
from optimizers import get_resnet18_optimizer
from schedulers import get_resnet18_scheduler
from dataloaders import dataloader
from criterion import build_criterion
from metrics import AverageMeter, accuracy
from model_utils import save_model, load_model

# External imports
import argparse 
from timm.data.mixup import Mixup
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)

def train_one_epoch(epoch, model, train_loader, gaussian_loader, criterion, optimizer, scheduler, 
                    config, scaler, mixup_fn):
    """
    Train for one epoch
    Adapted from CVT runner for ResNet18
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
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(
        name = f"resnet18-cifar10-{run_id}",
        reinit = True,
        project = "idl-project",
        config = cfg
    )
    gc.collect()
    torch.cuda.empty_cache()
    val_accs = []
    best_acc = -1
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
        is_best = (best_acc) == -1 or val_acc > best_acc
        best_acc = max(val_acc, best_acc)
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
    parser = argparse.ArgumentParser(description="ResNet18 Training Script")

    parser.add_argument('-d', '--dataset', type=str, default='cifar', help='Dataset: cifar')
    parser.add_argument('-n', '--run_name', type=str, required=True, help='Run name identifier')
    parser.add_argument('-l', '--lerac_epochs', type=int, default=5, help='LeRaC warmup epochs')
    parser.add_argument('-b', '--blur_epochs', type=int, default=20, help='Blur curriculum epochs')
    parser.add_argument('-c', '--c_factor', type=float, default=10.0, help='LeRaC growth factor C')
    parser.add_argument('-e', '--eta_min', type=float, default=2e-8, help='Minimum learning rate')
    args = parser.parse_args()
    
    run_logical_id = f'{args.dataset}_hybrid_{args.run_name}'
    
    if args.dataset.lower() == 'cifar':
        cfg = get_resnet18_cifar_config(run_id=run_logical_id, lerac_epochs=args.lerac_epochs, blur_epochs=args.blur_epochs, eta_min=args.eta_min)
        gaussian_dataloader = dataloader.build_gaussian_dataloader_cifar(cfg)
        train_loader, val_loader = dataloader.get_cifar_dataloaders(cfg)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    resnet18_model = resnet18.create_resnet18(cfg, num_classes=num_classes)
    params.count_parameters(resnet18_model)
    optimizer = get_resnet18_optimizer(cfg, resnet18_model)
    scheduler = get_resnet18_scheduler(cfg, optimizer, c_factor=args.c_factor)
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
    wandb_config['eta_min'] = args.eta_min
    do_training_loop(resnet18_model, wandb_config, train_loader, gaussian_dataloader, val_loader, criterion, criterion_eval, optimizer, scheduler, mixup_fn, run_id=run_logical_id)    
    model, optimizer, scheduler, epoch = load_model(resnet18_model, optimizer, scheduler, path=f"./OUTPUT/{run_logical_id}/best.pth")
    test_results = test(model, val_loader, cfg)

    print("Final Test Results")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Acc@1: {test_results['top1_acc']:.3f}%")

if __name__ == '__main__':
    main()

