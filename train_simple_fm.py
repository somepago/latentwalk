"""
Training pipeline for DiT flow matching on 32x32 shape images with DINO conditioning.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
import json
from datetime import datetime
import wandb

from datasets.shape_dataset import ShapeDataset
from models.dino import ModelWithIntermediateLayers
from models.dit import DiT
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

@torch.no_grad()
def generate_sample_images(dit, dino_model, device, save_dir, iteration, num_samples=4, num_steps=50):
    """
    Generate sample images using Euler sampling from the trained DiT.
    """
    dit.eval()

    # Create sample image prompts from the dataset
    dataset = ShapeDataset(num_samples=num_samples, image_size=64)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

    # Get sample images for conditioning
    sample_images, _ = next(iter(dataloader))
    sample_images = sample_images.to(device)

    # Get DINO features (resize to 56 for DINO ViT-S/14: 56/14 = 4x4 = 16 tokens)
    dino_images = torch.nn.functional.interpolate(
        sample_images, size=(56, 56), mode='bilinear', align_corners=False
    )
    dino_features = dino_model(dino_images).to(dtype=torch.float32)

    # Euler sampling: start from noise, integrate velocity
    x = torch.randn(num_samples, 3, 64, 64, device=device)
    dt = 1.0 / num_steps

    # Capture snapshots for denoising trajectory (first sample only)
    trajectory_steps = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4]
    trajectory_snapshots = [x[0].clamp(-1, 1).cpu()]

    for i in range(num_steps):
        t_val = 1.0 - i * dt  # go from t=1 (noise) to t=0 (clean)
        t = torch.full((num_samples,), t_val, device=device)
        v = dit(x, t, dino_features)
        x = x - v * dt  # velocity = noise - x_0, so we subtract to go toward x_0
        if i + 1 in trajectory_steps:
            trajectory_snapshots.append(x[0].clamp(-1, 1).cpu())

    generated = x.clamp(-1, 1)
    trajectory_snapshots.append(generated[0].cpu())  # final result

    # Save comparison grid
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        # Original images
        orig_img = sample_images[i].cpu().permute(1, 2, 0).numpy()
        orig_img = (orig_img + 1) / 2  # Denormalize from [-1,1] to [0,1]
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')

        # Generated images
        gen_img = generated[i].cpu().permute(1, 2, 0).numpy()
        gen_img = (gen_img + 1) / 2
        axes[1, i].imshow(gen_img)
        axes[1, i].set_title(f'Generated {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples', f'iter_{iteration}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save denoising trajectory for first sample
    n_snaps = len(trajectory_snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(n_snaps * 2.5, 3))
    labels = ['t=1.0', 't=0.75', 't=0.50', 't=0.25', 't=0.0', 'final'][:n_snaps]
    for i, (snap, label) in enumerate(zip(trajectory_snapshots, labels)):
        img = snap.permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        axes[i].imshow(img.clip(0, 1))
        axes[i].set_title(label)
        axes[i].axis('off')
    plt.tight_layout()
    traj_path = os.path.join(save_dir, 'samples', f'iter_{iteration}_trajectory.png')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated sample images for iteration {iteration}")
    dit.train()

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
    plot_interval=100,
    gradient_accumulation_steps=1,
    resume_path=None,
    wandb_project="latentwalk",
    wandb_entity=None,
    run_name=None,
):
    """
    Train the DiT model on shape dataset with flow matching.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {os.path.abspath(save_dir)}")

    # Initialize loss logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(save_dir, f'loss_log_{timestamp}.txt')
    loss_logger = LossLogger(log_file_path)
    print(f"Loss logging to: {log_file_path}")

    # Initialize dataset and dataloader (64x64 images)
    dataset = ShapeDataset(num_samples=num_samples, image_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize models
    dino_model = ModelWithIntermediateLayers().to(device)
    dino_model.eval()

    dit = DiT(image_size=64, patch_size=16).to(device)
    total_params = sum(p.numel() for p in dit.parameters())
    print(f"DiT parameters: {total_params:,}")

    # Optimizer and flat LR
    optimizer = AdamW(dit.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # Resume from checkpoint
    start_epoch = 0
    start_iteration = 0
    wandb_run_id = None
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        dit.load_state_dict(checkpoint['dit_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # Advance scheduler to the correct epoch
            for _ in range(checkpoint['epoch'] + 1):
                scheduler.step()
        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint.get('iteration', 0)
        wandb_run_id = checkpoint.get('wandb_run_id', None)
        print(f"Resumed at epoch {start_epoch}, iteration {start_iteration}")

    # Initialize wandb
    wandb_kwargs = dict(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_samples': num_samples,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'dit_params': total_params,
        },
    )
    if wandb_run_id is not None:
        wandb_kwargs['id'] = wandb_run_id
        wandb_kwargs['resume'] = 'must'
    wandb.init(**wandb_kwargs)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        num_batches = len(dataloader)
        timestep_losses = [[] for _ in range(4)]  # 4 buckets: [0,.25), [.25,.5), [.5,.75), [.75,1)

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_idx, (images, _) in enumerate(pbar):
                images = images.to(device)

                # Get DINO embeddings (resize 64 -> 56 for ViT-S/14)
                with torch.no_grad():
                    dino_images = torch.nn.functional.interpolate(
                        images, size=(56, 56), mode='bilinear', align_corners=False
                    )
                    dino_features = dino_model(dino_images).to(dtype=torch.float32)

                # Flow matching
                B = images.shape[0]
                t = torch.rand(B, device=device)
                noise = torch.randn_like(images)
                x_t = (1 - t[:, None, None, None]) * images + t[:, None, None, None] * noise
                target = noise - images  # velocity target

                # Forward pass
                pred = dit(x_t, t, dino_features)

                # Loss
                per_sample_loss = ((pred - target) ** 2).mean(dim=(1, 2, 3))
                loss = per_sample_loss.mean()

                # Loss by timestep bucket
                with torch.no_grad():
                    for bucket_idx, (lo, hi) in enumerate([(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]):
                        mask = (t >= lo) & (t < hi)
                        if mask.any():
                            timestep_losses[bucket_idx].append(per_sample_loss[mask].mean().item())

                loss = loss / gradient_accumulation_steps

                loss.backward()

                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # Log loss per iteration
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    iteration = epoch * (len(dataloader) // gradient_accumulation_steps) + (batch_idx // gradient_accumulation_steps) + 1
                    iter_loss = loss.item() * gradient_accumulation_steps
                    loss_logger.log_loss(iteration, epoch + 1, iter_loss)
                    log_dict = {
                        'loss': iter_loss,
                        'lr': scheduler.get_last_lr()[0],
                        'epoch': epoch + 1,
                        'grad_norm': grad_norm.item(),
                    }
                    bucket_names = ['loss_t0.00-0.25', 'loss_t0.25-0.50', 'loss_t0.50-0.75', 'loss_t0.75-1.00']
                    for bi, name in enumerate(bucket_names):
                        if timestep_losses[bi]:
                            log_dict[name] = sum(timestep_losses[bi]) / len(timestep_losses[bi])
                    timestep_losses = [[] for _ in range(4)]
                    wandb.log(log_dict, step=iteration)

                    # Generate sample images
                    if iteration % sample_interval == 0:
                        print(f"\nGenerating sample images for iteration {iteration}...")
                        generate_sample_images(dit, dino_model, device, save_dir, iteration)
                        sample_path = os.path.join(save_dir, 'samples', f'iter_{iteration}_comparison.png')
                        traj_path = os.path.join(save_dir, 'samples', f'iter_{iteration}_trajectory.png')
                        sample_log = {}
                        if os.path.exists(sample_path):
                            sample_log['samples'] = wandb.Image(sample_path)
                        if os.path.exists(traj_path):
                            sample_log['denoising_trajectory'] = wandb.Image(traj_path)
                        if sample_log:
                            wandb.log(sample_log, step=iteration)

                    # Save checkpoints
                    if iteration % save_interval == 0:
                        checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
                        print(f"\nSaving checkpoint to {checkpoint_path}")
                        torch.save({
                            'iteration': iteration,
                            'epoch': epoch,
                            'dit_state_dict': dit.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': total_loss / (batch_idx + 1),
                            'wandb_run_id': wandb.run.id if wandb.run else None,
                            'config': {
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'num_epochs': num_epochs,
                                'num_samples': num_samples,
                                'gradient_accumulation_steps': gradient_accumulation_steps
                            }
                        }, checkpoint_path)

                # Update progress bar
                total_loss += loss.item() * gradient_accumulation_steps
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1), "lr": scheduler.get_last_lr()[0]})

        scheduler.step()

    # Save final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f'checkpoint_final.pt')
    print(f"\nSaving final checkpoint to {final_checkpoint_path}")
    torch.save({
        'epoch': num_epochs - 1,
        'dit_state_dict': dit.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': total_loss / num_batches,
        'wandb_run_id': wandb.run.id if wandb.run else None,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_samples': num_samples,
            'gradient_accumulation_steps': gradient_accumulation_steps
        }
    }, final_checkpoint_path)

    # Generate final sample images
    print("Generating final sample images...")
    generate_sample_images(dit, dino_model, device, save_dir, 'final')

    # Generate final loss plots
    print("Generating final loss plots...")
    plot_loss_curves(log_file_path, save_dir, plot_interval)

    wandb.finish()
    print(f"Training completed! Loss log saved to: {log_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DiT on shape dataset with flow matching')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=100000)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--sample-interval', type=int, default=100)
    parser.add_argument('--plot-interval', type=int, default=100)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb-project', type=str, default='latentwalk')
    parser.add_argument('--wandb-entity', type=str, default=None)

    args = parser.parse_args()
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
        plot_interval=args.plot_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_path=args.resume,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )
