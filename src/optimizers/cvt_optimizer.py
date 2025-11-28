import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

CVT_BLOCKS = [1, 2, 10]
def get_cvt_optimizer(config, model, eta_0: float = .002):
    depth_prefixes = []
    for s, nb in enumerate(CVT_BLOCKS):
        depth_prefixes.append(f"stages.{s}.patch_embed")
        for b in range(nb):
            depth_prefixes.append(f"stages.{s}.blocks.{b}")

    param_names = [n for n, _ in model.named_parameters()]
    for extra in ["cls_token", "pos_embed", "norm", "head", "pre_logits", "fc"]:
        if any(n.startswith(extra) for n in param_names):
            depth_prefixes.append(extra)
    for s in range(len(CVT_BLOCKS)):  
        maybe = f"stages.{s}.downsample"
        if any(n.startswith(maybe) for n in param_names):
            depth_prefixes.append(maybe)

    rules = set(config['TRAIN']['WITHOUT_WD_LIST'])  
    no_decay_names = set()

    def is_depthwise(m: nn.Module):
        return isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels

    if {'bn','gn','ln'} & rules:
        for mod_name, mod in model.named_modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for p_name, _ in mod.named_parameters(recurse=False):
                    full = f"{mod_name}.{p_name}" if mod_name else p_name
                    no_decay_names.add(full)

    if 'bias' in rules:
        for n, p in model.named_parameters():
            if n.endswith(".bias"):
                no_decay_names.add(n)

    if 'dw' in rules:
        for mod_name, mod in model.named_modules():
            if is_depthwise(mod) and getattr(mod, "weight", None) is not None:
                no_decay_names.add(f"{mod_name}.weight")

    base_lr = float(config['TRAIN']['LR'])
    min_lr  = float(config['TRAIN']['LR_CURRICULUM']['MIN_LR'])

    init_lrs = np.logspace(np.log(base_lr), np.log(min_lr), len(depth_prefixes))

    buckets_decay    = OrderedDict((p, []) for p in depth_prefixes)
    buckets_nodecay  = OrderedDict((p, []) for p in depth_prefixes)
    misc_decay, misc_nodecay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        hit = False
        for pref in depth_prefixes:
            if n.startswith(pref):
                (buckets_nodecay[pref] if n in no_decay_names else buckets_decay[pref]).append(p)
                hit = True
                break
        if not hit:
            (misc_nodecay if n in no_decay_names else misc_decay).append(p)

    wd = float(config['TRAIN']['WD'])
    param_groups = []
    for pref, init_lr in zip(depth_prefixes, init_lrs):
        if buckets_decay[pref]:
            param_groups.append({'params': buckets_decay[pref],   'lr': init_lr, 'weight_decay': wd})
        if buckets_nodecay[pref]:
            param_groups.append({'params': buckets_nodecay[pref], 'lr': init_lr, 'weight_decay': 0.0})

    if misc_decay:
        param_groups.append({'params': misc_decay,   'lr': base_lr, 'weight_decay': wd,  '_init_lr': base_lr})
    if misc_nodecay:
        param_groups.append({'params': misc_nodecay, 'lr': base_lr, 'weight_decay': 0., '_init_lr': base_lr})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    print(f"[LeRaC] groups={len(param_groups)}  distinct_init_LRs={sorted({g['lr'] for g in param_groups}, reverse=True)[:6]}")
    return optimizer