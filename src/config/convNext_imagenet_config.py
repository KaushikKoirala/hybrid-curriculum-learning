from pathlib import Path

def get_convNext_imagenet_config(run_id: str = '', lerac_epochs: int = 5, blur_epochs: int = 20, eta_min: float = 1e-8):
    config = dict(
        num_epochs       = 100,          # keep it or change as you wish
        batch_size       = 128,          # tune to your GPU memory
        lr               = 4e-3,         # a good starting point for ImageNet-size data
        weight_decay     = 0.05,
        warmup_epochs    = 5,
        num_workers      = 8,            # ImageNet is stored on disk → use more workers
        image_size       = 224,
        amp              = True,
        pretrained       = False,

        dataset_path     = "ImageNet100_224",
        num_classes      = 100,
        ckpt_dir         = f"OUTPUT/{run_id}",

        lerac_epochs = lerac_epochs,
        blur_epochs = blur_epochs,
        eta_min     = eta_min,     # This is eta_n^0
    )
    Path(config["ckpt_dir"]).mkdir(exist_ok=True)

    return config
