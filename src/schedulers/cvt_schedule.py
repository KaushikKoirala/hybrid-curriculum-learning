from torch.optim.lr_scheduler import _LRScheduler, SequentialLR, CosineAnnealingLR

class LeRaCScheduler(_LRScheduler):
    """
    Implements the Learning Rate Curriculum (LeRaC) scheduler.

    This scheduler increases the learning rate of each parameter group
    from its initial value (eta_j^0) to a target value (eta^0) over
    a specified number of iterations (k)[cite: 15, 194].

    This scheduler should be stepped *every iteration*.

    Args:
        optimizer (Optimizer): The optimizer with LeRaC parameter groups.
        target_lr (float): The target learning rate (eta^0) that all groups
                           will reach at iteration k[cite: 190].
        num_iterations (int): The number of iterations (k) for the curriculum[cite: 196].
        c (float): The base for the exponential scheduler[cite: 203].
                   The paper fixes this at 10[cite: 203, 313].
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, target_lr, num_iterations, c=10.0, last_epoch=-1):
        self.target_lr = target_lr
        self.num_iterations = num_iterations
        self.c = c
        self.k = num_iterations

        # self.base_lrs stores the initial LRs (eta_j^0) for each group
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.last_epoch is the current *iteration* number (t)
        t = self.last_epoch

        # If curriculum is over, all LRs are the target_lr
        if t > self.k:
            return [self.target_lr for _ in self.base_lrs]

        new_lrs = []
        for eta_0_j in self.base_lrs: # eta_0_j is the initial LR for group j
            eta_k = self.target_lr

            # Avoid division by zero if eta_0_j is 0
            if eta_0_j == 0:
                new_lrs.append(0.0)
                continue

            # This implements Eq. 9: eta_j(t) = eta_j(0) * c^((t/k) * log_c(eta_k / eta_j(0)))
            #
            # log_ratio = np.log(eta_k / eta_0_j) / np.log(self.c)
            # exponent = (t / (self.k - 1.0)) * log_ratio
            # new_lr = eta_0_j * (self.c ** exponent)
            new_lr = eta_0_j * ((eta_k / eta_0_j) ** (t / self.k))
            new_lrs.append(new_lr)

        return new_lrs

def get_cvt_scheduler(config, optimizer, c_factor: float = 10.0):
    num_epochs = int(config['TRAIN']['END_EPOCH'])
    warmup_epochs = int(config['TRAIN']['LR_CURRICULUM'].get('WARMUP_EPOCHS', 5))
    base_lr = float(config['TRAIN']['LR'])
    min_lr = float(config['TRAIN']['LR_SCHEDULER']['ARGS'].get('min_lr', 1e-5))

    for g in optimizer.param_groups:
        if '_init_lr' not in g:
            g['_init_lr'] = g['lr']  

    lerac_scheduler = LeRaCScheduler(
        optimizer, target_lr=base_lr, num_iterations=warmup_epochs
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