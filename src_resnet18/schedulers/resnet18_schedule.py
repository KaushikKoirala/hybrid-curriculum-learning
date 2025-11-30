# Import LeRaC scheduler from local module
from .lerac_scheduler import LeRaCScheduler
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR


def get_resnet18_scheduler(config, optimizer, c_factor: float = 10.0):
    """
    Build LeRaC scheduler for ResNet18: Exponential warmup → Cosine annealing
    Uses the same LeRaC scheduler as CVT
    """
    num_epochs = int(config['TRAIN']['END_EPOCH'])
    warmup_epochs = int(config['TRAIN']['LR_CURRICULUM'].get('WARMUP_EPOCHS', 5))
    base_lr = float(config['TRAIN']['LR'])
    min_lr = float(config['TRAIN']['LR_SCHEDULER']['ARGS'].get('min_lr', 1e-5))

    # Store initial LRs for LeRaC warmup
    for g in optimizer.param_groups:
        if '_init_lr' not in g:
            g['_init_lr'] = g['lr']  

    lerac_scheduler = LeRaCScheduler(
        optimizer, target_lr=base_lr, num_iterations=warmup_epochs, c=c_factor
    )

    cosine_epochs = max(1, num_epochs - warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[lerac_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  
    )

    print(f"Scheduler: LeRaC warm-up ({warmup_epochs} ep) -> Cosine ({cosine_epochs} ep)")
    print(f"  Base LR: {base_lr} | Min LR: {min_lr} | Total epochs: {num_epochs}")
    return scheduler

