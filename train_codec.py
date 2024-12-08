import os
import time
from pathlib import Path
import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import wandb

from config.codec_config import CodecTrainingConfig
from tokenizer import WaveCodec, ResNetStack, GaussianZ
from dataset import AudioDataset
from utils import print_colored, tqdm0

def create_optimizer(model, config):
    # Prepare optimizer
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (T.nn.Linear, T.nn.Conv1d)
    blacklist_weight_modules = (T.nn.LayerNorm, T.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
                
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optim_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))
    return optimizer

def get_lr(step, config):
    # Cosine learning rate schedule with warmup
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    decay_steps = config.max_steps - config.warmup_steps
    step = min(step - config.warmup_steps, decay_steps)
    cosine_decay = 0.5 * (1 + T.cos(T.tensor(step/decay_steps * T.pi)))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine_decay

def train():
    config = CodecTrainingConfig()
    
    # Initialize wandb
    wandb.init(project="hertz-dev-codec-training", config=config.__dict__)
    
    # Create model
    print_colored("Creating codec model...", "blue")
    model = WaveCodec.Config(
        resnet_config=ResNetStack.Config(
            input_channels=config.input_channels,
            output_channels=config.output_channels,
            encode_channels=config.encode_channels,
            decode_channel_multiplier=config.decode_channel_multiplier,
            kernel_size=config.kernel_size,
            channel_ratios=config.channel_ratios,
            strides=config.strides,
            mode=None
        ),
        use_weight_norm=True,
        compressor_config=GaussianZ.Config(
            dim=None,
            latent_dim=config.latent_dim,
            bias=True,
            use_weight_norm=True
        ),
        norm_stddev=0.05
    )()
    
    if config.compile:
        model = T.compile(model)
    
    model = model.to(config.device)
    
    # Create datasets and dataloaders
    print_colored("Creating datasets...", "blue")
    train_dataset = AudioDataset(config.train_data_path, config.sample_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    scaler = GradScaler()
    
    # Training loop
    print_colored("Starting training...", "green")
    step = 0
    while step < config.max_steps:
        model.train()
        for batch in tqdm0(train_loader):
            # Update learning rate
            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                
            audio = batch.to(config.device)
            
            # Forward pass
            with autocast(dtype=T.bfloat16):
                reconstructed, latent = model(audio)
                
                # Calculate losses
                recon_loss = F.mse_loss(reconstructed, audio)
                kl_loss = latent['kl_divergence']
                
                loss = config.reconstruction_weight * recon_loss + config.kl_weight * kl_loss
                
                # Scale loss by grad accumulation
                loss = loss / config.grad_accum_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                T.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if step % config.log_interval == 0:
                wandb.log({
                    "loss": loss.item() * config.grad_accum_steps,
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "lr": lr,
                    "step": step,
                })
                
            # Save checkpoint
            if step % config.save_interval == 0:
                ckpt_path = os.path.join(config.checkpoint_dir, f"step_{step}.pt")
                T.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, ckpt_path)
                
            step += 1
            if step >= config.max_steps:
                break
                
    print_colored("Training complete!", "green")
    wandb.finish()

if __name__ == "__main__":
    train()
