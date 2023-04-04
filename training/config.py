import torch

LR = 1e-3
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
CONTEXT_SIZE = 10
BATCH_SIZE=32

model_config = {
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
      }
    },
    "zero_optimization": {
      "stage": 3,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 1e9,
      "reduce_bucket_size": "auto",
      "stage3_prefetch_bucket_size": "auto",
      "stage3_param_persistence_threshold": "auto",
      "stage3_max_live_parameters": 1e9,
      "stage3_max_reuse_distance": 1e9,
      "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "wall_clock_breakdown": False
  }