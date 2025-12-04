def get_resnet18_imagenet_config(run_id: str = '', lerac_epochs: int = 5, blur_epochs: int = 20, eta_min: float = 2e-8):
    """
    Configuration for ResNet-18 on ImageNet with hybrid curriculum learning
    """
    config = {
        "OUTPUT_DIR": f"OUTPUT/{run_id}",
        "WORKERS": 8,
        "PRINT_FREQ": 500,
        "AMP": {
            "ENABLED": True
        },
        "MODEL": {
            "NAME": "resnet18",
            "NUM_CLASSES": 1000
        },
        "AUG": {
            "MIXUP_PROB": 1.0,
            "MIXUP": 0.8,
            "MIXCUT": 1.0,
            "TIMM_AUG": {
                "USE_LOADER": True,
                "RE_COUNT": 1,
                "RE_MODE": "pixel",
                "RE_SPLIT": False,
                "RE_PROB": 0.25,
                "AUTO_AUGMENT": "rand-m9-mstd0.5-inc1",
                "HFLIP": 0.5,
                "VFLIP": 0.0,
                "COLOR_JITTER": 0.4,
                "INTERPOLATION": "bicubic"
            }
        },
        "LOSS": {
            "LABEL_SMOOTHING": 0.1
        },
        "CUDNN": {
            "BENCHMARK": True,
            "DETERMINISTIC": False,
            "ENABLED": True
        },
        "DATASET": {
            "DATASET": "imagenet",
            "DATA_FORMAT": "jpg",
            "ROOT": "./ImageNet100_224",
            "TEST_SET": "val",
            "TRAIN_SET": "train"
        },
        "TEST": {
            "BATCH_SIZE_PER_GPU": 256,
            "IMAGE_SIZE": [224, 224],
            "MODEL_FILE": "",
            "INTERPOLATION": "bicubic"
        },
        "TRAIN": {
            "BATCH_SIZE_PER_GPU": 256,
            "GRADIENT_ACCUMULATION_STEPS": 1,       
            "LR": 0.01,  # Base learning rate for ResNet-18
            "IMAGE_SIZE": [224, 224],
            "BEGIN_EPOCH": 0,
            "END_EPOCH": 100,
            "LR_CURRICULUM": {
                "MIN_LR": eta_min,
                "WARMUP_EPOCHS": lerac_epochs,
                "ETA_0": 0.1  # Learning rate decay factor within layers (η^(0)=0.1)
            },
            "LR_SCHEDULER": {
                "METHOD": "lerac",
                "ARGS": {
                    "sched": "cosine",
                    "warmup_epochs": 5,
                    "warmup_lr": 1e-5,
                    "min_lr": 1e-6,  # Minimum LR for cosine phase
                    "cooldown_epochs": 0,
                    "decay_rate": 0.1
                }
            },
            "OPTIMIZER": "adamW",
            "WD": 5e-4,  # Weight decay for ResNet-18
            "WITHOUT_WD_LIST": ["bn", "bias", "ln"],
            "CLF_LR_MULTIPLIER": 0.01,  # Classifier LR multiplier
            "SHUFFLE": True,
            "BLUR": {
                "KERNEL_SIZE": 7,
                "SIGMA": 1.0,
                "EPOCHS": blur_epochs
            }      
        },
        "DEBUG": {
            "DEBUG": False
        }
    }
    return config

