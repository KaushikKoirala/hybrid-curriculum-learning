def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*60)
    print("Model Parameter Count")
    print("="*60)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("="*60)
    
    return total_params, trainable_params