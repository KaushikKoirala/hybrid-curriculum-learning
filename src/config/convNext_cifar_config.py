from pathlib import Path

def get_convNext_cifar_config(run_id: str = '', lerac_epochs: int = 5, blur_epochs: int = 20, eta_min: float = 1e-8):
    config = dict(
        num_epochs       = 100,
        batch_size       = 128,
        lr               = 4e-3,
        weight_decay     = 0.05,
        warmup_epochs    = 5,
        num_workers      = 8,
        image_size       = 224,          # ConvNeXt default
        val_split_ratio  = 0.1,
        amp              = True,         # Automatic Mixed Precision
        pretrained       = False,

        dataset_path     = "./cifar-10",
        num_classes      = 10,
        ckpt_dir         = f"OUTPUT/{run_id}",

        lerac_epochs = lerac_epochs,
        blur_epochs = blur_epochs,
        eta_min     = eta_min,     # This is eta_n^0
    )
    
    Path(config["ckpt_dir"]).mkdir(exist_ok=True)

    return config
