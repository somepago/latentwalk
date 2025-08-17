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
from models.diffusion_utils import FlowMatchingScheduler, DPMSolver
from models.vae_utils import load_vae, vae_encode, vae_decode
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def generate_sample_images(model, vae, dino_model, projector, device, save_dir, epoch, num_samples=4):
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
    for i, img in enumerate(pil_images):
        img.save(os.path.join(save_dir, 'samples', f'epoch_{epoch}_sample_{i}.png'))
    
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
    plt.savefig(os.path.join(save_dir, 'samples', f'epoch_{epoch}_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated sample images for epoch {epoch}")
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
    print(f"Save directory: {os.path.abspath(save_dir)}")

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

    projector = Projector().to(device, dtype=model_dtype)
    
    sana_model = Sana_600M(
        in_channels=in_channels,
        hidden_size=768,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qk_norm=True,
        y_norm=True,
    ).to(device, dtype=model_dtype)

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
                    'num_samples': num_samples,
                    'use_vae': vae is not None,
                    'in_channels': in_channels
                }
            }, checkpoint_path)
        
        # Generate sample images (independent of checkpoint saving)
        if (epoch + 1) % sample_interval == 0:
            print(f"Generating sample images for epoch {epoch + 1}...")
            generate_sample_images(sana_model, vae, dino_model, projector, device, save_dir, epoch + 1)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(save_dir, f'checkpoint_final.pt')
    print(f"\nSaving final checkpoint to {final_checkpoint_path}")
    torch.save({
        'epoch': num_epochs - 1,
        'projector_state_dict': projector.state_dict(),
        'sana_state_dict': sana_model.state_dict(),
        'projector_optimizer_state_dict': projector_optimizer.state_dict(),
        'sana_optimizer_state_dict': sana_optimizer.state_dict(),
        'loss': total_loss / num_batches,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_samples': num_samples,
            'use_vae': vae is not None,
            'in_channels': in_channels
        }
    }, final_checkpoint_path)
    
    # Generate final sample images
    print("Generating final sample images...")
    generate_sample_images(sana_model, vae, dino_model, projector, device, save_dir, 'final')

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
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save checkpoint every N epochs (default: 10)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--sample-interval', type=int, default=10,
                        help='generate sample images every N epochs (default: 10)')

    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        device=args.device,
        num_samples=args.num_samples,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        sample_interval=args.sample_interval
    )
