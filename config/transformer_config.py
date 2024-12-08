from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrainingConfig:
    # Model architecture
    dim: int = 4096
    n_head: int = 32
    n_layer: int = 32
    vocab_size: int = 32768
    seq_len: int = 2048
    
    # Training params
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    
    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Data
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    sample_rate: int = 16000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
