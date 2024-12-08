from dataclasses import dataclass
from typing import Optional

@dataclass
class CodecTrainingConfig:
    # Model architecture 
    input_channels: int = 1
    output_channels: int = 1
    encode_channels: int = 16
    decode_channel_multiplier: int = 4
    kernel_size: int = 7
    channel_ratios: tuple = (4, 8, 16, 16, 16, 16)
    strides: tuple = (2, 2, 4, 5, 5, 5)
    latent_dim: int = 32
    
    # Training params
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.01
    
    # Optimizer
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    
    # Data
    train_data_path: str = "data/train"
    val_data_path: str = "data/val" 
    sample_rate: int = 16000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints/codec"
    
    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
