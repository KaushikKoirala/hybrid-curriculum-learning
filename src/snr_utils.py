import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class FeatureExtractor:
    """ Helper class to retrieve intermediate feature maps. """
    def __init__(self, model, target_layers):
        self.model = model
        self.features = {}
        self.hooks = []
        
        # Register hooks
        for name, module in model.named_modules():
            if name in target_layers:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def calculate_layer_snr(model_current, model_converged, dataloader, device, num_batches=5):
    """
    Calculates SNR per layer by comparing current features to converged features.
    
    SNR formula used: 10 * log10( Power_Signal / Power_Noise )
    Where:
      - Signal = Converged Feature Map
      - Noise = (Current Feature Map - Converged Feature Map)
    """
    model_current.eval()
    model_converged.eval()
    
    target_layers = [
        "features.0", 
        "features.1", 
        "features.3", 
        "features.5", 
        "features.7"
    ]
    layer_names = ["Stem", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    # Attach hooks
    extractor_curr = FeatureExtractor(model_current, target_layers)
    extractor_conv = FeatureExtractor(model_converged, target_layers)

    layer_snrs = {name: [] for name in target_layers}

    print("Calculating SNR across layers...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches: break
            
            images = images.to(device)
            
            # Forward pass
            _ = model_current(images)
            _ = model_converged(images)
            
            # Calculate SNR for each layer
            for layer in target_layers:
                feat_curr = extractor_curr.features[layer]
                feat_conv = extractor_conv.features[layer]
                
                # Signal Power: Variance of the converged (clean) features
                # We use mean of squares approx for power
                signal_power = torch.mean(feat_conv ** 2)
                
                # Noise Power: Variance of the difference
                noise = feat_curr - feat_conv
                noise_power = torch.mean(noise ** 2)
                
                # Avoid division by zero
                if noise_power == 0:
                    noise_power = 1e-8
                
                # SNR in dB
                snr = 10 * torch.log10(signal_power / noise_power)
                layer_snrs[layer].append(snr.item())

    # Average across batches
    avg_snrs = [np.mean(layer_snrs[layer]) for layer in target_layers]
    
    # Cleanup
    extractor_curr.remove_hooks()
    extractor_conv.remove_hooks()
    
    return layer_names, avg_snrs

def visualize_snr(layer_names, snr_values, save_path="snr_plot.png"):
    """
    Plots the SNR decay curve similar to Figure 4 in the LeRaC paper.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(layer_names, snr_values, marker='o', linestyle='-', linewidth=2, color='teal')
    
    # Add value annotations
    for i, val in enumerate(snr_values):
        plt.text(i, val + 0.5, f"{val:.2f}", ha='center', fontweight='bold')

    plt.title("Layer-wise Signal-to-Noise Ratio (SNR)", fontsize=14)
    plt.ylabel("10 x log(SNR) [dB]", fontsize=12)
    plt.xlabel("Layer Depth", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label="Noise Floor (Signal = Noise)")
    plt.legend()
    
    plt.savefig(save_path)
    print(f"SNR Plot saved to {save_path}")
    plt.close()