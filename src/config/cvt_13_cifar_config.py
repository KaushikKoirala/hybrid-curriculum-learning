def get_cvt_13_cifar_config(run_id: str = '', lerac_epochs: int = 5, blur_epochs: int = 20):
    # params largely copied from https://github.com/leoxiaobin/CvT/blob/main/experiments/imagenet/cvt/cvt-13-224x224.yaml
    config = {
    "OUTPUT_DIR": f"OUTPUT/{run_id}",
    "WORKERS": 8,
    "PRINT_FREQ": 500,
    "AMP": {
        "ENABLED": True
    },
    "MODEL": {
        "NAME": "cls_cvt",
        "SPEC": {
        "INIT": "trunc_norm",
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 2, 10],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1]
        }
    },
    "AUG": {
        "MIXUP_PROB": 1.0,
        "MIXUP": 0.8,
        "MIXCUT": 1.0,
        "TIMM_AUG": {
        "USE_LOADER": False,
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
        "DATASET": "cifar-10",
        "DATA_FORMAT": "jpg",
        "ROOT": "./cifar-10",
        "TEST_SET": "val",
        "TRAIN_SET": "train"
    },
    "TEST": {
        "BATCH_SIZE_PER_GPU": 256,
        "IMAGE_SIZE": [32, 32],
        "MODEL_FILE": "",
        "INTERPOLATION": "bicubic"
    },
    "TRAIN": {
        "BATCH_SIZE_PER_GPU": 512,
        "GRADIENT_ACCUMULATION_STEPS": 4,       
        "LR": .00125,
        "IMAGE_SIZE": [32, 32],
        "BEGIN_EPOCH": 0,
        "END_EPOCH": 100,
        "LR_CURRICULUM": {
            "MIN_LR": 2e-6, #https://github.com/CroitoruAlin/LeRaC/blob/main/experiments/cvt_experiments.py#L78
            "WARMUP_EPOCHS": lerac_epochs
        },
        "LR_SCHEDULER": {
        "METHOD": "timm",
        "ARGS": {
            "sched": "cosine",
            "warmup_epochs": 5,
            "warmup_lr": 0.000001,
            "min_lr": 0.00001,
            "cooldown_epochs": 10,
            "decay_rate": 0.1
        }
        },
        "OPTIMIZER": "adamW",
        "WD": 0.05,
        "WITHOUT_WD_LIST": ["bn", "bias", "ln"],
        "SHUFFLE": True,
        "BLUR": {
            "KERNEL_SIZE": 5,
            "SIGMA": 1,
            "EPOCHS": blur_epochs
        }      
    },
    "DEBUG": {
        "DEBUG": False
    }
    }
    return config
