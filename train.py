"""
Training pipeline for finetuning projector and SANA model on shape images.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from datetime import datetime

from datasets.shape_dataset import ShapeDataset
from models.dino import ModelWithIntermediateLayers
from models.projector import Projector, CrossAttentionProjector
from models.diffusion_model import Sana_600M
from models.diffusion_utils import FlowMatchingScheduler, DPMSolver
from models.vae_utils import load_vae, vae_encode, vae_decode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class LossLogger:
    """Logger for tracking training loss per iteration."""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.losses = []
        self.iterations = []
        self.epochs = []
        
        # Create log file and write header
        with open(self.log_file_path, 'w') as f:
            f.write("iteration,epoch,loss\n")
    
    def log_loss(self, iteration, epoch, loss):
        """Log a single loss value."""
        self.iterations.append(iteration)
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # Write to file immediately
        with open(self.log_file_path, 'a') as f:
            f.write(f"{iteration},{epoch},{loss:.6f}\n")
    
    def get_recent_losses(self, num_points=100):
        """Get the most recent loss values for plotting."""
        if len(self.losses) <= num_points:
            return self.iterations, self.losses
        else:
            return self.iterations[-num_points:], self.losses[-num_points:]

def plot_loss_curves(log_file_path, save_dir, plot_interval=100):
    """
    Plot loss curves from the logged data and save as images.
    
    Args:
        log_file_path (str): Path to the loss log file
        save_dir (str): Directory to save the plots
        plot_interval (int): How often to save plots (every N iterations)
    """
    if not os.path.exists(log_file_path):
        print(f"Loss log file not found: {log_file_path}")
        return
    
    # Read the log file
    iterations = []
    epochs = []
    losses = []
    
    with open(log_file_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            if line.strip():
                iter_num, epoch, loss = line.strip().split(',')
                iterations.append(int(iter_num))
                epochs.append(int(epoch))
                losses.append(float(loss))
    
    if not losses:
        print("No loss data found in log file")
        return
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'loss_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Loss vs Iteration (full training)
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, losses, 'b-', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_vs_iteration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Loss vs Iteration (recent window)
    if len(losses) > plot_interval:
        recent_iterations = iterations[-plot_interval:]
        recent_losses = losses[-plot_interval:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(recent_iterations, recent_losses, 'r-', alpha=0.8, linewidth=1.5)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training Loss vs Iteration (Last {plot_interval} iterations)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'loss_vs_iteration_recent.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Average Loss per Epoch
    epoch_losses = {}
    for epoch, loss in zip(epochs, losses):
        if epoch not in epoch_losses:
            epoch_losses[epoch] = []
        epoch_losses[epoch].append(loss)
    
    avg_epoch_losses = {epoch: np.mean(losses) for epoch, losses in epoch_losses.items()}
    epoch_numbers = sorted(avg_epoch_losses.keys())
    avg_losses = [avg_epoch_losses[epoch] for epoch in epoch_numbers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, avg_losses, 'g-', marker='o', markersize=4, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_loss_per_epoch.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {plots_dir}")

def generate_sample_images(model, vae, dino_model, projector, device, save_dir, iteration, num_samples=4):
    """
    Generate sample images using the current model state.
    """
    model.eval()
    
    # Create sample image prompts from the dataset
    dataset = ShapeDataset(num_samples=num_samples, image_size=1024, min_size=128, max_size=256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)
    
    # Get sample images
    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images.to(device)
    
    # Resize for DINO
    dino_images = torch.nn.functional.interpolate(
        sample_images, 
        size=(224, 224), 
        mode='bilinear', 
        align_corners=False
    )
    
    with torch.no_grad():
        # Get DINO features
        dino_features = dino_model(dino_images)
        dino_features = dino_features.to(dtype=model.dtype)
        
        # Project to SANA condition space
        projected_features, mask = projector(dino_features)
        
        # Create scheduler and sampler
        scheduler = FlowMatchingScheduler(num_train_timesteps=1000, shift=3.0)
        sampler = DPMSolver(
            model=model,
            scheduler=scheduler,
            num_inference_steps=20,
            guidance_scale=4.5,
            flow_shift=3.0,
        )
        
        # Generate latents
        latents = sampler.sample(
            text_embeddings=projected_features,
            attention_mask=mask,
            null_embeddings=projected_features,  # Use same embeddings for null
            height=32,
            width=32,
            generator=None,
            device=device,
        )
        
        # Decode latents to images
        if vae is not None:
            images = vae_decode("AutoencoderDC", vae, latents)
            # Convert to PIL images
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype(np.uint8)
            pil_images = [Image.fromarray(img) for img in images]
        else:
            # Fallback: visualize latents directly
            pil_images = []
            for i in range(latents.shape[0]):
                latent = latents[i].mean(dim=0)
                latent = latent.cpu().numpy()
                # Resize for visualization
                from scipy.ndimage import zoom
                latent = zoom(latent, (32, 32), order=1)
                latent = (latent * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(np.stack([latent] * 3, axis=-1)))
    
    # Save images
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    # for i, img in enumerate(pil_images):
    #     img.save(os.path.join(save_dir, 'samples', f'epoch_{epoch}_sample_{i}.png'))
    
    # Create a grid visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i in range(num_samples):
        # Original images
        orig_img = sample_images[i].cpu().permute(1, 2, 0).numpy()
        orig_img = (orig_img + 1) / 2  # Denormalize
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Generated images
        axes[1, i].imshow(pil_images[i])
        axes[1, i].set_title(f'Generated {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples', f'iter_{iteration}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated sample images for iteration {iteration}")
    model.train()

def train(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    save_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_samples=100000,
    save_interval=10,
    num_workers=4,
    sample_interval=10,
    projector_type='linear',
    plot_interval=100,
    train_type='projector',
    pretrained_sana_path=None,
    gradient_accumulation_steps=1,
):
    """
    Train the SANA model on shape dataset.
    
    Args:
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizers
        num_epochs (int): Number of epochs to train
        save_dir (str): Directory to save model checkpoints
        device (str): Device to use for training ('cuda' or 'cpu')
        num_samples (int): Number of samples in the training dataset
        save_interval (int): Save checkpoint every N epochs
        projector_type (str): Type of projector to use (linear or crossattention)
        train_type (str): Type of params to train (projector or both)
        pretrained_sana_path (str): Path to pre-trained SANA model for projector-only training
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {os.path.abspath(save_dir)}")

    # Initialize loss logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(save_dir, f'loss_log_{timestamp}.txt')
    loss_logger = LossLogger(log_file_path)
    print(f"Loss logging to: {log_file_path}")

    # Initialize dataset and dataloader
    # SANA needs 1024x1024 images, DINO needs multiples of 14 (224x224 is good)
    dataset = ShapeDataset(num_samples=num_samples, image_size=1024, min_size=128, max_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize models
    dino_model = ModelWithIntermediateLayers().to(device)
    dino_model.eval()  # DINO model is frozen
    
    # Load VAE for encoding images to latents
    vae = load_vae(device=device, dtype=torch.float32)  # Use float32 for stability
    if vae is None:
        print("Warning: VAE not available, falling back to pixel training")
        in_channels = 3
        model_dtype = torch.float32
    else:
        print("VAE loaded successfully, training on latents")
        in_channels = 32  # DC-AE VAE uses 32 channels
        model_dtype = torch.float32  # Use float32 for stability

    if projector_type == 'linear':
        projector = Projector().to(device, dtype=model_dtype)
    elif projector_type == 'crossattention':
        projector = CrossAttentionProjector().to(device, dtype=model_dtype)
    else:
        raise ValueError(f"Invalid projector type: {projector_type}")
    
    sana_model = Sana_600M().to(device, dtype=model_dtype)

    # Load pre-trained SANA model if specified
    if pretrained_sana_path is not None and os.path.exists(pretrained_sana_path):
        print(f"Loading pre-trained SANA model from {pretrained_sana_path}")
        checkpoint = torch.load(pretrained_sana_path, map_location=device)
        
        # Handle different checkpoint formats like in inference.py
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "sana_state_dict" in checkpoint:
            state_dict = checkpoint["sana_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = sana_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            print(f"First few missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            print(f"First few unexpected keys: {unexpected_keys[:5]}...")
            
        print("Pre-trained SANA model loaded successfully")
    else:
        print("‼️⚠️ Warning: SANA weights not found, initializing from scratch")

    # Initialize optimizers
    projector_optimizer = AdamW(projector.parameters(), lr=learning_rate)
    if train_type == 'both':
        sana_optimizer = AdamW(sana_model.parameters(), lr=learning_rate)
    else:
        sana_model.requires_grad_(False)
        sana_optimizer = None

    # Initialize scheduler
    flow_scheduler = FlowMatchingScheduler()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(device)

                # Get DINO embeddings
                with torch.no_grad():
                    # Resize images to 224x224 for DINO (multiple of 14)
                    dino_images = torch.nn.functional.interpolate(
                        images, 
                        size=(224, 224), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    dino_features = dino_model(dino_images)
                    # Convert DINO features to model dtype
                    dino_features = dino_features.to(dtype=model_dtype)

                # Project DINO features to SANA condition space
                projected_features, mask = projector(dino_features)

                # Encode images to latents if VAE is available
                if vae is not None:
                    with torch.no_grad():
                        # Ensure images are in the correct dtype for VAE
                        vae_dtype = next(vae.parameters()).dtype
                        images_for_vae = images.to(dtype=vae_dtype)
                        # Encode 1024x1024 images to latents using DC-AE VAE
                        latents = vae_encode("AutoencoderDC", vae, images_for_vae, sample_posterior=False, device=device)
                    # Use latents for training
                    training_data = latents
                else:
                    # Use pixel images for training
                    training_data = images

                # Sample random timesteps
                batch_size = training_data.shape[0]
                t = torch.rand(batch_size, device=device, dtype=model_dtype)

                # Get noisy samples and targets using flow matching
                # Generate noise with same shape as training data
                noise = torch.randn_like(training_data)
                # Get noisy samples
                noisy_samples = flow_scheduler.add_noise(training_data, noise, t)
                # Get velocity targets
                targets = flow_scheduler.get_velocity(training_data, noise, t)

                # Get model predictions
                pred = sana_model(noisy_samples, t, projected_features, mask=mask)

                # Calculate loss
                loss = nn.MSELoss()(pred, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backpropagation
                loss.backward()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    projector_optimizer.step()
                    projector_optimizer.zero_grad()
                    if sana_optimizer is not None:
                        sana_optimizer.step()
                        sana_optimizer.zero_grad()

                # Log loss per iteration (only count as iteration when we actually update)
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    iteration = epoch * (len(dataloader) // gradient_accumulation_steps) + (batch_idx // gradient_accumulation_steps) + 1
                    loss_logger.log_loss(iteration, epoch + 1, loss.item() * gradient_accumulation_steps)  # Scale back for logging

                    # Generate sample images based on iteration count
                    if iteration % sample_interval == 0:
                        print(f"\nGenerating sample images for iteration {iteration}...")
                        generate_sample_images(sana_model, vae, dino_model, projector, device, save_dir, f'iter_{iteration}')

                    # Save checkpoints based on iteration count
                    if iteration % save_interval == 0:
                        checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
                        print(f"\nSaving checkpoint to {checkpoint_path}")
                        checkpoint_data = {
                            'iteration': iteration,
                            'epoch': epoch,
                            'projector_state_dict': projector.state_dict(),
                            'sana_state_dict': sana_model.state_dict(),
                            'projector_optimizer_state_dict': projector_optimizer.state_dict(),
                            'loss': total_loss / num_batches,
                            'config': {
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'num_epochs': num_epochs,
                                'num_samples': num_samples,
                                'use_vae': vae is not None,
                                'in_channels': in_channels,
                                'gradient_accumulation_steps': gradient_accumulation_steps
                            }
                        }
                        
                        # Only save SANA optimizer state if it exists
                        if sana_optimizer is not None:
                            checkpoint_data['sana_optimizer_state_dict'] = sana_optimizer.state_dict()
                        
                        torch.save(checkpoint_data, checkpoint_path)

                # Update progress bar
                total_loss += loss.item() * gradient_accumulation_steps  # Scale back for display
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            

        

    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f'checkpoint_final.pt')
    print(f"\nSaving final checkpoint to {final_checkpoint_path}")
    
    final_checkpoint_data = {
        'epoch': num_epochs - 1,
        'projector_state_dict': projector.state_dict(),
        'sana_state_dict': sana_model.state_dict(),
        'projector_optimizer_state_dict': projector_optimizer.state_dict(),
        'loss': total_loss / num_batches,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_samples': num_samples,
            'use_vae': vae is not None,
            'in_channels': in_channels,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }
    }
    
    # Only save SANA optimizer state if it exists
    if sana_optimizer is not None:
        final_checkpoint_data['sana_optimizer_state_dict'] = sana_optimizer.state_dict()
    
    torch.save(final_checkpoint_data, final_checkpoint_path)
    
    # Generate final sample images
    print("Generating final sample images...")
    generate_sample_images(sana_model, vae, dino_model, projector, device, save_dir, 'final')
    
    # Generate final loss plots
    print("Generating final loss plots...")
    plot_loss_curves(log_file_path, save_dir, plot_interval)
    
    print(f"Training completed! Loss log saved to: {log_file_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SANA model on shape dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='directory to save model checkpoints (default: checkpoints)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='number of samples in the training dataset (default: 100000)')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='save checkpoint every N iterations (default: 1000)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--sample-interval', type=int, default=100,
                        help='generate sample images every N iterations (default: 100)')
    # projector type
    parser.add_argument('--projector-type', type=str, default='linear', choices=['linear', 'crossattention'],
                        help='type of projector to use (default: linear)')
    # which params to train - projector only or projector and sana
    parser.add_argument('--train-type', type=str, default='projector', choices=['projector', 'both'],
                        help='type of params to train (default: projector)')
    # loss plot interval
    parser.add_argument('--plot-interval', type=int, default=100,
                        help='number of iterations to include in recent loss plot (default: 100)')
    # give the run a name
    parser.add_argument('--run-name', type=str, default=None,
                        help='name of the run (default: run)')
    # pretrained SANA model path for projector-only training
    parser.add_argument('--pretrained-sana-path', type=str, default="checkpoints/sana_600m.pt",
                        help='path to pre-trained SANA model for projector-only training')
    # gradient accumulation
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='number of steps to accumulate gradients before updating (default: 1)')


    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.projector_type, args.train_type)
    if args.run_name is not None:
        args.save_dir = os.path.join(args.save_dir, args.run_name)
    
    train(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        device=args.device,
        num_samples=args.num_samples,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        sample_interval=args.sample_interval,
        projector_type=args.projector_type,
        train_type=args.train_type,
        plot_interval=args.plot_interval,
        pretrained_sana_path=args.pretrained_sana_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
