"""
Training pipeline for finetuning projector and SANA model on shape images.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

from datasets.shape_dataset import ShapeDataset
from models.dino import ModelWithIntermediateLayers
from models.projector import Projector
from models.diffusion_model import Sana_600M
from models.diffusion_utils import FlowMatchingScheduler

def train(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    save_dir="checkpoints",
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_samples=100000,
    save_interval=10,
    num_workers=4,
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
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = ShapeDataset(num_samples=num_samples, image_size=224, min_size=32, max_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize models
    dino_model = ModelWithIntermediateLayers().to(device)
    dino_model.eval()  # DINO model is frozen

    projector = Projector().to(device)
    sana_model = Sana_600M(
        in_channels=3,
        hidden_size=768,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qk_norm=True,
        y_norm=True,
    ).to(device)

    # Initialize optimizers
    projector_optimizer = AdamW(projector.parameters(), lr=learning_rate)
    sana_optimizer = AdamW(sana_model.parameters(), lr=learning_rate)

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
                    dino_features = dino_model(images)

                # Project DINO features to SANA condition space
                projected_features, mask = projector(dino_features)

                # Sample random timesteps
                batch_size = images.shape[0]
                t = torch.rand(batch_size, device=device)

                # Get noisy samples and targets using flow matching
                # Generate noise
                noise = torch.randn_like(images)
                # Get noisy samples
                noisy_samples = flow_scheduler.add_noise(images, noise, t)
                # Get velocity targets
                targets = flow_scheduler.get_velocity(images, noise, t)

                # Get model predictions
                pred = sana_model(noisy_samples, t, projected_features, mask=mask)

                # Calculate loss
                loss = nn.MSELoss()(pred, targets)

                # Backpropagation
                projector_optimizer.zero_grad()
                sana_optimizer.zero_grad()
                loss.backward()
                projector_optimizer.step()
                sana_optimizer.step()

                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            print(f"\nSaving checkpoint to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'projector_state_dict': projector.state_dict(),
                'sana_state_dict': sana_model.state_dict(),
                'projector_optimizer_state_dict': projector_optimizer.state_dict(),
                'sana_optimizer_state_dict': sana_optimizer.state_dict(),
                'loss': total_loss / num_batches,
                'config': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'num_samples': num_samples
                }
            }, checkpoint_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SANA model on shape dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--save-dir', type=str, default='outputs',
                        help='directory to save model checkpoints (default: outputs)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='number of samples in the training dataset (default: 100000)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save checkpoint every N epochs (default: 10)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')

    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        device=args.device
    )
