import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


def _is_depthwise(m):
    """Check if module is depthwise convolution"""
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels
        and m.groups == m.out_channels
    )


def set_wd(config, model):
    """Separate parameters by weight decay"""
    without_decay_list = config.get('WITHOUT_WD_LIST', ['bn', 'bias', 'ln'])
    without_decay_depthwise = []
    without_decay_norm = []

    for m in model.modules():
        if _is_depthwise(m) and 'dw' in without_decay_list:
            without_decay_depthwise.append(m.weight)
        elif isinstance(m, nn.BatchNorm2d) and 'bn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.GroupNorm) and 'gn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.LayerNorm) and 'ln' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)

    with_decay = []
    without_decay = []

    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    skip_keys = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keys = model.no_weight_decay_keywords()

    for n, p in model.named_parameters():
        ever_set = False

        if p.requires_grad is False:
            continue

        skip_flag = False
        if n in skip:
            without_decay.append(p)
            skip_flag = True
        else:
            for i in skip:
                if i in n:
                    without_decay.append(p)
                    skip_flag = True

        if skip_flag:
            continue

        for i in skip_keys:
            if i in n:
                skip_flag = True

        if skip_flag:
            continue

        for pp in without_decay_depthwise:
            if p is pp:
                without_decay.append(p)
                ever_set = True
                break

        for pp in without_decay_norm:
            if p is pp:
                without_decay.append(p)
                ever_set = True
                break

        if (
            (not ever_set)
            and 'bias' in without_decay_list
            and n.endswith('.bias')
        ):
            without_decay.append(p)
        elif not ever_set:
            with_decay.append(p)

    params = [
        {'params': with_decay},
        {'params': without_decay, 'weight_decay': 0.}
    ]
    return params


def get_resnet18_optimizer(config, model):
    """
    Build optimizer with LeRaC (Learning Rate Curriculum) for ResNet-18
    Based on the notebook implementation
    """
    base_lr = float(config['TRAIN']['LR'])
    min_lr = float(config['TRAIN']['LR_CURRICULUM']['MIN_LR'])
    weight_decay = float(config['TRAIN']['WD'])
    eta_0 = config['TRAIN']['LR_CURRICULUM'].get('ETA_0', 0.1)  # η^(0) = 0.1
    clf_lr_multiplier = config['TRAIN'].get('CLF_LR_MULTIPLIER', 0.01)

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    blocks = [str(i) for i in range(2)]  # ResNet-18 has 2 blocks per layer

    # Build learning rate dictionary with LeRaC decay factor η^(0) = 0.1
    dictionary_lr = {}
    current_lr = base_lr  # Start with base_lr for first layer/block
    init_lrs = np.logspace(np.log(base_lr), np.log(min_lr), len(layers))
    idx = 0

    for layer in layers:
        for block in blocks:
            key = f"{layer}.{block}"
            current_lr = init_lrs[idx]
            dictionary_lr[key] = max(current_lr, min_lr)
        idx += 1

    # Get parameter groups with weight decay handling
    params_with_wd_info = set_wd(config['TRAIN'], model)

    # Build parameter groups with layer-wise learning rates
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    new_param_groups = []

    # Process both with_decay and without_decay groups
    for group in params_with_wd_info:
        group_params = group['params'] if isinstance(group['params'], list) else [group['params']]
        wd = group.get('weight_decay', weight_decay)

        for param in group_params:
            name = param_to_name.get(id(param), "unknown")
            assigned_lr = base_lr

            # Check if parameter belongs to a specific layer block
            if name.startswith('conv1') or name.startswith('bn1'):
                assigned_lr = base_lr
            #elif 'fc' in name or 'linear' in name:
            #    assigned_lr = base_lr * clf_lr_multiplier
            else:
                # Check layer blocks
                for key in dictionary_lr:
                    if name.startswith(key):
                        assigned_lr = dictionary_lr[key]
                        break

            new_param_groups.append({
                'params': param,
                'lr': assigned_lr,
                'weight_decay': wd
            })

    optimizer = torch.optim.AdamW(new_param_groups, betas=(0.9, 0.999), eps=1e-8)

    print("=" * 80)
    print(f"Optimizer: AdamW with LeRaC for ResNet-18")
    print(f"Total parameter groups: {len(new_param_groups)}")
    print(f"Base LR: {base_lr:.10f}")
    print(f"Min LR: {min_lr:.10f}")
    print(f"Weight Decay: {weight_decay}")
    print("=" * 80)
    print("Layer-wise Learning Rates:")
    print("-" * 80)
    print(f"conv1/bn1:       {base_lr:.6f}")
    for layer in layers:
        for block in blocks:
            key = f"{layer}.{block}"
            if key in dictionary_lr:
                print(f"{key:15s} {dictionary_lr[key]:.6f}")
    #print(f"classifier:      {base_lr * clf_lr_multiplier:.6f}")
    print("-" * 80)

    return optimizer


