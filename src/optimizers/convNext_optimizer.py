import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def get_convnext_tiny_lerac_groups(model, base_lr, lerac_end_lr):
    """
    Creates parameter groups for ConvNextTiny with LeRaC initial LRs.

    Args:
        model (nn.Module): The ConvNextTiny model.
        base_lr (float): The base learning rate (eta^0), for the first layer[cite: 210, 313].
        lerac_end_lr (float): The final learning rate (eta_n^0), for the last layer[cite: 210].
    """

    # Define the logical "layers" of ConvNextTiny from input to output
    # This is a manual process based on the model architecture.
    layers = []

    # 1. Features (Backbone)
    if hasattr(model, 'features'):
        layers.extend([
            model.features[0],  # Stem
            model.features[1],  # Stage 1
            model.features[2],  # Downsample
            model.features[3],  # Stage 2
            model.features[4],  # Downsample
            model.features[5],  # Stage 3
            model.features[6],  # Downsample
            model.features[7],  # Stage 4
        ])
    else:
        print("Warning: 'model.features' not found. Check model architecture.")

    # 2. Classifier Head
    if hasattr(model, 'classifier'):
        # Add the LayerNorm before the head
        if isinstance(model.classifier[0], nn.LayerNorm):
             layers.append(model.classifier[0])

        # Add the final Linear layer (user-replaced)
        if isinstance(model.classifier[-1], nn.Linear):
            layers.append(model.classifier[-1])
        else:
            print(f"Warning: Expected nn.Linear at model.classifier[-1], but found {type(model.classifier[-1])}.")
    else:
         print("Warning: 'model.classifier' not found. Check model architecture.")

    num_layers = len(layers)
    if num_layers == 0:
        print("Error: No layers were found. Returning default parameter group.")
        return model.parameters()

    print(f"LeRaC: Found {num_layers} logical layers for parameter groups.")

    # Generate the initial learning rates (linear interpolation in log-space)
    # [eta_1^0, ..., eta_n^0]
    initial_lerac_lrs = np.logspace(
        np.log10(base_lr),
        np.log10(lerac_end_lr),
        num_layers
    )

    param_groups = []
    for layer, lr in zip(layers, initial_lerac_lrs):
        param_groups.append({
            'params': layer.parameters(),
            'lr': lr
        })

    return param_groups


def get_convNextT_optimizer(config, model):
  # Create LeRaC Parameter Groups
  print("Setting up LeRaC parameter groups...")
  param_groups = get_convnext_tiny_lerac_groups(
      model,
      config["lr"],
      config["lerac_end_lr"]
  )

  # Create Optimizer
  optimizer = optim.AdamW(
      param_groups,
      lr=config["lr"], # This default lr is ignored
      weight_decay=config["weight_decay"]
  )
  return optimizer