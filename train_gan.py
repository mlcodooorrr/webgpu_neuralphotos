import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from logger import TrainingLogger
import torchvision
from dataloader import distributed_data_loader
from gan import GAN
from discriminator import SimpleDiscriminator, discriminator_loss, generator_adversarial_loss, gradient_penalty as gp_loss_fn
import argparse
from datetime import datetime
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

from torch.amp import autocast


# ==================== CONFIGURATION ====================
class Config:
    batch_size = 1024
    num_epochs = 400
    batches_per_epoch = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimizer
    lr_g = 2e-4
    lr_d = 2e-5
    betas = (0.5, 0.999)  # beta1=0.5 is standard for GANs
    
    # Loss weights
    lambda_l1 = 100.0   # Keep L1 dominant
    lambda_adv = 1.0    # Set to 0 for L1-only training
    lambda_gp = 10.0    # Gradient penalty weight
    
    loss_type = 'hinge'
    
    # Training stability
    clip_grad = 1.0
    d_updates_per_g = 1  # Train D and G equally
    
    # Validation
    val_interval = 10
    val_batches = 1
    
    # Logging
    save_interval = 25

    # Input noise
    use_input_noise = False
    input_noise_level = 0.10
    input_noise_probability = 1.0  # Add noise 100% of the time

    # Scheduled Sampling
    use_multistep_sampling = True
    sampling_schedule = {
        0: 0,       # Pure L1 - learn basic reconstruction
        30: 1,      # Introduce 1 generated frame
        60: 2,
        100: 4,
        150: 8,
        200: 16,
        260: 32,    # Full autoregressive
    }
    
    sampling_probability_schedule = {
        0: 0.0,
        30: 0.2,
        60: 0.3,
        100: 0.4,
        150: 0.5,
        200: 0.6,
        260: 0.7,
    }

def get_sampling_probability(epoch, probability_schedule):
    """Get sampling probability for current epoch"""
    prob = 0.0
    for epoch_threshold, p in sorted(probability_schedule.items()):
        if epoch >= epoch_threshold:
            prob = p
        else:
            break
    return prob

def generate_noise(device, B):
    noise_0 = torch.rand(B, 1, 4, 3, device=device)
    noise_1 = torch.rand(B, 1, 16, 12, device=device)
    noise_2 = torch.rand(B, 1, 64, 48, device=device)
    noise_3 = torch.rand(B, 1, 256, 192, device=device)
    return noise_0, noise_1, noise_2, noise_3

def get_num_generated_frames(epoch, sampling_schedule):
    """
    Determine how many frames to replace with generated ones based on epoch
    """
    num_frames = 0
    for epoch_threshold, frames in sorted(sampling_schedule.items()):
        if epoch >= epoch_threshold:
            num_frames = frames
        else:
            break
    return num_frames

def add_noise_to_inputs(past_0, past_1, past_2, past_3, noise_level=0.01):
    """
    Add small noise to input frames to improve robustness
    This makes the model more robust to imperfect inputs
    """
    past_0 = past_0 + torch.randn_like(past_0) * noise_level
    past_1 = past_1 + torch.randn_like(past_1) * noise_level
    past_2 = past_2 + torch.randn_like(past_2) * noise_level
    past_3 = past_3 + torch.randn_like(past_3) * noise_level
    
    # Clamp to keep in valid range (model expects [-1, 1])
    past_0 = torch.clamp(past_0, -1.0, 1.0)
    past_1 = torch.clamp(past_1, -1.0, 1.0)
    past_2 = torch.clamp(past_2, -1.0, 1.0)
    past_3 = torch.clamp(past_3, -1.0, 1.0)
    
    return past_0, past_1, past_2, past_3

def inference_stability_check(G, val_loader, config, num_frames=100):
    """
    Run autoregressive inference and check if outputs stay stable.
    Returns stats dict.
    """
    G.eval()
    
    # Get one batch for initial memory
    inputs, targets = next(iter(val_loader))
    
    controls = inputs['controls'][0:1].to(device)  # Just 1 sample
    past_0 = inputs['past_0'][0:1].to(device)
    past_1 = inputs['past_1'][0:1].to(device)
    past_2 = inputs['past_2'][0:1].to(device)
    past_3 = inputs['past_3'][0:1].to(device)
    
    means = []
    stds = []
    
    with torch.no_grad():
        for i in range(num_frames):
            noise_0, noise_1, noise_2, noise_3 = generate_noise(device, 1)
            
            generated = G(controls, past_0, past_1, past_2, past_3,
                         noise_0, noise_1, noise_2, noise_3)
            
            means.append(generated.mean().item())
            stds.append(generated.std().item())
            
            # Update memory
            past_3 = torch.cat([past_3[:, 1:], generated.unsqueeze(1)], dim=1)
            
            gen_64x48 = F.interpolate(generated, size=(64, 48), mode='area')
            past_2 = torch.cat([past_2[:, 1:], gen_64x48.unsqueeze(1)], dim=1)
            
            gen_16x12 = F.interpolate(generated, size=(16, 12), mode='area')
            past_1 = torch.cat([past_1[:, 1:], gen_16x12.unsqueeze(1)], dim=1)
            
            gen_4x3 = F.interpolate(generated, size=(4, 3), mode='area')
            past_0 = torch.cat([past_0[:, 1:], gen_4x3.unsqueeze(1)], dim=1)
    
    G.train()
    
    return {
        'mean_start': means[0],
        'mean_end': means[-1],
        'mean_drift': abs(means[-1] - means[0]),
        'std_start': stds[0],
        'std_end': stds[-1],
        'std_drift': abs(stds[-1] - stds[0]),
    }

def generate_autoregressive_memory(G, controls, past_0, past_1, past_2, past_3, 
                                   num_steps, device):
    """
    Generate multiple frames autoregressively to replace memory buffer
    
    Args:
        G: Generator model
        controls: (B, 16) control vector
        past_0, past_1, past_2, past_3: Initial memory buffers
        num_steps: How many frames to generate
        device: torch device
    
    Returns:
        Updated memory buffers with generated frames
    """
    B = controls.shape[0]
    
    with torch.no_grad():
        for step in range(num_steps):
            # Generate fresh noise for each step
            noise_0 = torch.rand(B, 1, 4, 3, device=device)
            noise_1 = torch.rand(B, 1, 16, 12, device=device)
            noise_2 = torch.rand(B, 1, 64, 48, device=device)
            noise_3 = torch.rand(B, 1, 256, 192, device=device)
            
            # Generate frame
            generated = G(controls, past_0, past_1, past_2, past_3,
                         noise_0, noise_1, noise_2, noise_3)
            
            # Update past_3 (last 4 frames at full resolution)
            past_3 = torch.cat([past_3[:, 1:], generated.unsqueeze(1)], dim=1)
            
            # Update past_2 (last 8 frames at 64x48)
            gen_64x48 = F.interpolate(generated, size=(64, 48), mode='area')
            past_2 = torch.cat([past_2[:, 1:], gen_64x48.unsqueeze(1)], dim=1)
            
            # Update past_1 (last 16 frames at 16x12)
            gen_16x12 = F.interpolate(generated, size=(16, 12), mode='area')
            past_1 = torch.cat([past_1[:, 1:], gen_16x12.unsqueeze(1)], dim=1)
            
            # Update past_0 (all 32 frames at 4x3)
            gen_4x3 = F.interpolate(generated, size=(4, 3), mode='area')
            past_0 = torch.cat([past_0[:, 1:], gen_4x3.unsqueeze(1)], dim=1)
    
    return past_0, past_1, past_2, past_3

def train_one_epoch(G, D, optimizer_G, optimizer_D, train_loader, epoch, config, logger, run_dir):
    """Train for one epoch - GAN with L1"""

    # nvtx.range_push(f"Epoch {epoch}")

    G.train()
    if config.lambda_adv > 0:
        D.train()
    
    # Track metrics
    d_losses = []
    g_adv_losses = []
    g_l1_losses = []
    g_total_losses = []
    gp_losses = []
    
    pbar = tqdm(range(config.batches_per_epoch), desc=f"Epoch {epoch}")
    train_iter = iter(train_loader)

    num_generated_frames = get_num_generated_frames(epoch, config.sampling_schedule)
    sampling_probability = get_sampling_probability(epoch, config.sampling_probability_schedule)
    
    for batch_idx in pbar:
        # Get batch
        inputs, targets = next(train_iter)
        # inputs, targets = cast_inputs_to_bf16(inputs, targets)

        # TODO: miniscule, unpack inputs once instead of two access calls 
        # TODO: my pin_memory is false, nonblocking isn't really doing anything here, non_blocking lets transfer start but doesn't have to finish, so next operations can start
        # TODO: pin memory slows down computation a bit for some reason
        # TODO: dtype bfloat16 might need to be changed back to float32
        # inputs = {k: v.to(device, dtype=torch.bfloat16, non_blocking=True) for k,v in inputs.items()}
        # targets = targets.to(device, dtype=torch.bfloat16, non_blocking=True)

        controls = inputs['controls'].to(device, non_blocking=True)
        past_0 = inputs['past_0'].to(device, non_blocking=True)
        past_1 = inputs['past_1'].to(device, non_blocking=True)
        past_2 = inputs['past_2'].to(device, non_blocking=True)
        past_3 = inputs['past_3'].to(device, non_blocking=True)
        real_frames = targets.to(device, non_blocking=True)

        # Generate noise on GPU (instead of loading from dataloader)
        B = targets.shape[0]
        noise_0, noise_1, noise_2, noise_3 = generate_noise(device, B)

        # Input noise augmentation (disabled by default now)
        if config.use_input_noise and torch.rand(1).item() < config.input_noise_probability:
            past_0, past_1, past_2, past_3 = add_noise_to_inputs(
                past_0, past_1, past_2, past_3, 
                noise_level=config.input_noise_level
            )
        
        # Scheduled sampling - use probability schedule
        used_scheduled_sampling = False
        if config.use_multistep_sampling and num_generated_frames > 0:
            if torch.rand(1).item() < sampling_probability:
                used_scheduled_sampling = True
                past_0, past_1, past_2, past_3 = generate_autoregressive_memory(
                    G, controls, past_0, past_1, past_2, past_3,
                    num_generated_frames, device
                )
        
        # Train discriminator (if adversarial loss enabled)
        d_loss = None
        gp = None
        real_preds = None
        fake_preds = None
        
        if config.lambda_adv > 0:
            for _ in range(config.d_updates_per_g):
                optimizer_D.zero_grad()
                
                # Generate fake frames (detach to not update G)
                with torch.no_grad():
                    fake_frames = G(controls, past_0, past_1, past_2, past_3,
                                noise_0, noise_1, noise_2, noise_3)
                
                with autocast(device_type=config.device, dtype=torch.float32, enabled=False):
                    # Get discriminator predictions
                    real_preds = D(real_frames)
                    fake_preds = D(fake_frames.detach())
                    
                    # Compute discriminator loss
                    d_loss = discriminator_loss(real_preds, fake_preds, loss_type=config.loss_type)
                    
                    # Gradient penalty
                    gp = gp_loss_fn(D, real_frames, fake_frames, device=config.device)
                    d_total_loss = d_loss + config.lambda_gp * gp
                
                # Backward and update
                d_total_loss.backward()
                if config.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), config.clip_grad)
                optimizer_D.step()
        
        # Train generator
        optimizer_G.zero_grad()
        
        with autocast(device_type=config.device, dtype=torch.float32, enabled=False):
            # Generate fake frames (with gradients this time)
            fake_frames = G(controls, past_0, past_1, past_2, past_3,
                        noise_0, noise_1, noise_2, noise_3)
            
            # Compute generator losses
            if config.lambda_adv > 0:
                # Get discriminator's opinion on fakes
                fake_preds_for_g = D(fake_frames)
                g_adv_loss = generator_adversarial_loss(fake_preds_for_g, loss_type=config.loss_type)
                g_l1_loss = F.l1_loss(fake_frames, real_frames)
                g_total_loss = config.lambda_adv * g_adv_loss + config.lambda_l1 * g_l1_loss
            else:
                # L1-only training
                g_adv_loss = torch.tensor(0.0)
                g_l1_loss = F.l1_loss(fake_frames, real_frames)
                g_total_loss = config.lambda_l1 * g_l1_loss
        
        # Backward and update
        g_total_loss.backward()
        if config.clip_grad > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), config.clip_grad)
        optimizer_G.step()
        
        # ========== TRACK METRICS ==========
        batch_metrics = {
            'd_loss': d_loss.item() if d_loss is not None else 0.0,
            'g_adv_loss': g_adv_loss.item() if isinstance(g_adv_loss, torch.Tensor) else 0.0,
            'g_l1_loss': g_l1_loss.item(),
            'g_total_loss': g_total_loss.item(),
            'gp_loss': gp.item() if gp is not None else 0.0,
            'grad_norm_g': grad_norm.item() if config.clip_grad > 0 else 0.0,
        }
        
        d_losses.append(batch_metrics['d_loss'])
        g_adv_losses.append(batch_metrics['g_adv_loss'])
        g_l1_losses.append(batch_metrics['g_l1_loss'])
        g_total_losses.append(batch_metrics['g_total_loss'])
        gp_losses.append(batch_metrics['gp_loss'])
        
        # LOG BATCH METRICS
        logger.log_batch(epoch, batch_idx, batch_metrics)

        
        # if batch_idx == 5:
        #     profiler.start()
        # if batch_idx == 10:
        #     profiler.stop()
        #     break

        
        # Detailed logging every 100 batches
        if batch_idx % 100 == 0:
            print(f"\n📊 Batch {batch_idx} - Epoch {epoch}:")
            print(f"  Generator output range: [{fake_frames.min().item():.3f}, {fake_frames.max().item():.3f}]")
            print(f"  G L1: {g_l1_loss.item():.4f}")
            print(f"  Scheduled Sampling: {num_generated_frames} frames @ {sampling_probability:.0%} prob | This batch: {'YES' if used_scheduled_sampling else 'NO'}")
            
            # ✅ NEW: Show scheduled sampling info
            if num_generated_frames > 0:
                print(f"  🔄 Scheduled Sampling: {num_generated_frames} generated frames")
            
            if config.lambda_adv > 0:
                print(f"  D loss: {d_loss.item():.4f} | GP: {gp.item():.4f}")
                print(f"  G adv: {g_adv_loss.item():.4f}")
                print(f"  Real preds: {real_preds.mean().item():.3f} | Fake preds: {fake_preds.mean().item():.3f}")
            
            # Save sample images to run directory
            sample_fake = fake_frames[0].detach().cpu()
            sample_real = real_frames[0].detach().cpu()
            sample_fake = (sample_fake + 1) / 2  # Denormalize
            sample_real = (sample_real + 1) / 2
            
            # Save side by side
            comparison = torch.cat([sample_real, sample_fake], dim=2)  # Concatenate horizontally
            img_path = os.path.join(run_dir, f'sample_epoch{epoch}_batch{batch_idx}.png')
            torchvision.utils.save_image(comparison, img_path)
            print(f"  💾 Saved comparison to {img_path}")
        
        # Update progress bar
        if config.lambda_adv > 0:
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G_adv': f'{g_adv_loss.item():.3f}',
                'G_L1': f'{g_l1_loss.item():.3f}',
                'GenFrames': num_generated_frames  # ✅ NEW
            })
        else:
            pbar.set_postfix({
                'G_L1': f'{g_l1_loss.item():.3f}',
                'GenFrames': num_generated_frames  # ✅ NEW
            })
    
    # nvtx.range_pop()

    return {
        'd_loss': sum(d_losses) / len(d_losses) if d_losses else 0.0,
        'g_adv_loss': sum(g_adv_losses) / len(g_adv_losses) if g_adv_losses else 0.0,
        'g_l1_loss': sum(g_l1_losses) / len(g_l1_losses),
        'g_total_loss': sum(g_total_losses) / len(g_total_losses),
        'gp_loss': sum(gp_losses) / len(gp_losses) if gp_losses else 0.0,
    }


def validate(G, D, val_loader, config):
    """Validate the model - GAN with L1"""
    G.eval()
    if config.lambda_adv > 0:
        D.eval()
    
    d_losses = []
    g_adv_losses = []
    g_l1_losses = []
    g_total_losses = []
    
    val_iter = iter(val_loader)
    
    with torch.no_grad():
        for _ in tqdm(range(config.val_batches), desc="Validating"):
            # Get batch
            inputs, targets = next(val_iter)
            
            controls = inputs['controls'].to(device, non_blocking=True)
            past_0 = inputs['past_0'].to(device, non_blocking=True)
            past_1 = inputs['past_1'].to(device, non_blocking=True)
            past_2 = inputs['past_2'].to(device, non_blocking=True)
            past_3 = inputs['past_3'].to(device, non_blocking=True)
            real_frames = targets.to(device, non_blocking=True)

            B = targets.shape[0]
            noise_0, noise_1, noise_2, noise_3 = generate_noise(device, B)

            with autocast(device_type=config.device, dtype=torch.float32, enabled=False):
                # Generate fake frames
                fake_frames = G(controls, past_0, past_1, past_2, past_3,
                            noise_0, noise_1, noise_2, noise_3)
                
                # Compute losses
                g_l1_loss = F.l1_loss(fake_frames, real_frames)
                
                if config.lambda_adv > 0:
                    # Discriminator predictions
                    real_preds = D(real_frames)
                    fake_preds = D(fake_frames)
                    
                    d_loss = discriminator_loss(real_preds, fake_preds, loss_type=config.loss_type)
                    g_adv_loss = generator_adversarial_loss(fake_preds, loss_type=config.loss_type)
                    g_total_loss = config.lambda_adv * g_adv_loss + config.lambda_l1 * g_l1_loss
                else:
                    d_loss = torch.tensor(0.0)
                    g_adv_loss = torch.tensor(0.0)
                    g_total_loss = config.lambda_l1 * g_l1_loss
            
            # Track metrics
            d_losses.append(d_loss.item())
            g_adv_losses.append(g_adv_loss.item())
            g_l1_losses.append(g_l1_loss.item())
            g_total_losses.append(g_total_loss.item())
    
    return {
        'd_loss': sum(d_losses) / len(d_losses) if d_losses else 0.0,
        'g_adv_loss': sum(g_adv_losses) / len(g_adv_losses) if g_adv_losses else 0.0,
        'g_l1_loss': sum(g_l1_losses) / len(g_l1_losses),
        'g_total_loss': sum(g_total_losses) / len(g_total_losses),
    }


def save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, config, run_dir, filename):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'config': {
            'lr_g': config.lr_g,
            'lr_d': config.lr_d,
            'lambda_l1': config.lambda_l1,
            'lambda_adv': config.lambda_adv,
            'lambda_gp': config.lambda_gp,
            'loss_type': config.loss_type,
        }
    }, filepath)
    
    print(f"✅ Checkpoint saved: {filepath}")


def load_checkpoint(G, D, optimizer_G, optimizer_D, filepath):
    """Load model checkpoint, handling compiled models"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # ✅ NEW: Handle compiled model checkpoints (strip _orig_mod. prefix)
    g_state_dict = checkpoint['generator_state_dict']
    d_state_dict = checkpoint['discriminator_state_dict']
    
    # Remove _orig_mod. prefix if present (from torch.compile)
    g_state_dict = {k.replace('_orig_mod.', ''): v for k, v in g_state_dict.items()}
    d_state_dict = {k.replace('_orig_mod.', ''): v for k, v in d_state_dict.items()}
    
    # Load state dicts
    G.load_state_dict(g_state_dict)
    D.load_state_dict(d_state_dict)
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"✅ Checkpoint loaded from epoch {epoch}")
    
    return epoch


def weights_init_generator(m):
    """Weight initialization for generator"""
    classname = m.__class__.__name__
    
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    
    elif classname == 'ConvBlock':
        nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
        if hasattr(m.conv, 'bias') and m.conv.bias is not None:
            nn.init.constant_(m.conv.bias.data, 0.0)


def weights_init_discriminator(m):
    """Weight initialization for discriminator"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GAN/L1 model')
    parser.add_argument('--name', type=str, default=None, 
                        help='Name for this training run (default: auto-generated timestamp)')
    args = parser.parse_args()

    
    # Create run directory
    if args.name is None:
        # Auto-generate name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    else:
        run_name = args.name
        
    run_dir = os.path.join("training_runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Run directory: {run_dir}")

    config = Config()
    device = config.device
    batch_size = config.batch_size

    print("="*70)
    if config.lambda_adv > 0:
        print("🎯 GAN TRAINING MODE (L1 + Adversarial)")
    else:
        print("🎯 L1-ONLY TRAINING MODE")
    print("="*70)
    
    # ========== SETUP MODELS ==========
    print("Setting up models...")
    # Generator
    G = GAN().to(device)
    # Discriminator
    D = SimpleDiscriminator().to(device)

    # ========== SETUP OPTIMIZERS ==========
    optimizer_G = torch.optim.AdamW(G.parameters(), lr=config.lr_g, betas=config.betas)
    optimizer_D = torch.optim.AdamW(D.parameters(), lr=config.lr_d, betas=config.betas)

    # ========== SETUP CHECKPOINT PATH ==========
    start_epoch = 0
    checkpoint_path = f"training_runs/{run_name}/checkpoints/latest.pth"

    print("Loading pretrained L1 generator...")
    if os.path.exists(checkpoint_path):
        print(f"📂 Found checkpoint: {checkpoint_path}")
        start_epoch = load_checkpoint(G, D, optimizer_G, optimizer_D, checkpoint_path)
        start_epoch += 1  # Start from next epoch
        print(f"🔄 Resuming training from epoch {start_epoch}")
    else:
        print("⚠️ No checkpoint found, starting fresh")
        G.apply(weights_init_generator)
        if config.lambda_adv > 0:
            D.apply(weights_init_discriminator)

    # TODO: fix converting model to torch.memory channels last, skip the conversions with convolutions, fix bfloat16 conversion
    # G = G.to(dtype=torch.bfloat16)
    # G = torch.compile(G, mode='max-autotune')
    
    # Print number of parameters
    g_num_params = sum(p.numel() for p in G.parameters())
    print(f"Number of parameters in Generator: {g_num_params:,}")
    if config.lambda_adv > 0:
        d_num_params = sum(p.numel() for p in D.parameters())
        print(f"Number of parameters in Discriminator: {d_num_params:,}")

    # ========== SETUP BATCHES PER EPOCH ==========
    if config.batches_per_epoch is None:
        train_samples = int(25198 * 0.9)  # 90% train split
        config.batches_per_epoch = train_samples // config.batch_size
        print(f"📊 Batches per epoch: {config.batches_per_epoch} (full dataset pass)")
    
    # ========== SETUP DATA ==========
    train_loader = distributed_data_loader(B=batch_size, split="train", device=device)
    val_loader = distributed_data_loader(B=batch_size, split="val", device=device)
    
    inputs, targets = next(iter(train_loader))    

    # ========== SETUP LOGGER ==========
    log_file = os.path.join(run_dir, "training_log.json")
    logger = TrainingLogger(log_file)
    logger.log_config(config)
    print(f"📝 Logging to: {log_file}")
    
    # ========== TRAINING LOOP ==========    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*70}")
        if config.lambda_adv > 0:
            print(f"Epoch {epoch}/{config.num_epochs} - GAN Training")
        else:
            print(f"Epoch {epoch}/{config.num_epochs} - L1 Training")
        print(f"{'='*70}")

        # if epoch == 1:
        #     profiler.start()

        # Train
        train_metrics = train_one_epoch(G, D, optimizer_G, optimizer_D, train_loader, epoch, config, logger, run_dir)

        # if epoch == 3:
        #     profiler.stop()
        #     break

        # # # Exit after first epoch when profiling
        # print("\n✅ Profiling complete! Exiting.")
        # break

        logger.log_train_epoch(epoch, train_metrics)

        print(f"\n📊 Training Summary:")
        if config.lambda_adv > 0:
            print(f"  D loss:      {train_metrics['d_loss']:.4f}")
            print(f"  G adv loss:  {train_metrics['g_adv_loss']:.4f}")
            print(f"  GP loss:     {train_metrics['gp_loss']:.4f}")
        print(f"  G L1 loss:   {train_metrics['g_l1_loss']:.4f}")
        print(f"  G total:     {train_metrics['g_total_loss']:.4f}")

        # Validate
        if (epoch + 1) % config.val_interval == 0:
            print(f"\n🔍 Running validation...")
            val_metrics = validate(G, D, val_loader, config)

            # Inference stability check
            print(f"\n🔬 Running inference stability check...")
            stability = inference_stability_check(G, val_loader, config, num_frames=100)
            print(f"  Mean: {stability['mean_start']:.3f} → {stability['mean_end']:.3f} (drift: {stability['mean_drift']:.3f})")
            print(f"  Std:  {stability['std_start']:.3f} → {stability['std_end']:.3f} (drift: {stability['std_drift']:.3f})")
            
            # Warn if drifting badly
            if stability['mean_drift'] > 0.1 or stability['std_drift'] > 0.1:
                print(f"  ⚠️ WARNING: Significant drift detected!")

            logger.log_val(epoch, val_metrics)
            
            print(f"\n📊 Validation Summary:")
            if config.lambda_adv > 0:
                print(f"  D loss:      {val_metrics['d_loss']:.4f}")
                print(f"  G adv loss:  {val_metrics['g_adv_loss']:.4f}")
            print(f"  G L1 loss:   {val_metrics['g_l1_loss']:.4f}")
            print(f"  G total:     {val_metrics['g_total_loss']:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, config, run_dir, f'epoch_{epoch}.pth')
        
        # Always save latest
        save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, config, run_dir, 'latest.pth')

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"📁 Results saved to: {run_dir}")
    print("=" * 70)