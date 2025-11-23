from torch.optim.lr_scheduler import _LRScheduler, SequentialLR, CosineAnnealingLR

class LeRaCScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_epochs, c=10.0, last_epoch = -1):
        self.base_lr = float(base_lr)
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.c = float(c)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # After warmup: all groups share base_lr
        if self.last_epoch >= self.warmup_epochs:
            return [self.base_lr for _ in self.optimizer.param_groups]

        # Warmup epoch counter t \in {1..warmup_epochs}
        t = self.last_epoch + 1

        lrs = []
        for g in self.optimizer.param_groups:
            init = float(g.get('_init_lr', g['lr']))
            # growth per Eq.(9), clamped to base
            lr_t = init * (self.c ** t)
            lrs.append(self.base_lr if lr_t >= self.base_lr else lr_t)
        return lrs

def get_cvt_scheduler(config, optimizer, c_factor: float = 10.0):
    num_epochs = int(config['TRAIN']['END_EPOCH'])
    warmup_epochs = int(config['TRAIN']['LR_CURRICULUM'].get('WARMUP_EPOCHS', 5))
    base_lr = float(config['TRAIN']['LR'])
    min_lr = float(config['TRAIN']['LR_SCHEDULER']['ARGS'].get('min_lr', 1e-5))

    for g in optimizer.param_groups:
        if '_init_lr' not in g:
            g['_init_lr'] = g['lr']  

    lerac_scheduler = LeRaCScheduler(
        optimizer, base_lr=base_lr, warmup_epochs=warmup_epochs, last_epoch=-1, c=c_factor
    )

    cosine_epochs = max(1, num_epochs - warmup_epochs-1)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[lerac_scheduler, cosine_scheduler],
        milestones=[warmup_epochs+1]  
    )

    print(f"Scheduler: LeRaC warm-up ({warmup_epochs + 1} ep) -> Cosine ({cosine_epochs} ep)")
    print(f"  Base LR: {base_lr} | Min LR: {min_lr} | Total epochs: {num_epochs}")
    return scheduler