import torch
import matplotlib.pyplot as plt
from datasets.shape_dataset import ShapeDataset
import numpy as np
from PIL import Image
import random


def plot_interpolation_grid(dataset, idx1, idx2, alphas):
    """
    Create a grid of interpolated images showing both pixel and parameter interpolation.
    """
    n_alphas = len(alphas)
    fig, axes = plt.subplots(2, n_alphas + 2, figsize=(3*(n_alphas + 2), 6))
    
    # Get the original images
    img1, meta1 = dataset[idx1]
    img2, meta2 = dataset[idx2]
    
    # Plot original images
    axes[0, 0].imshow(img1[0].numpy(), cmap='gray')
    axes[0, 0].set_title(f'Source\n{dataset.shape_types[meta1["shape_type"]]}')
    axes[1, 0].imshow(img1[0].numpy(), cmap='gray')
    
    axes[0, -1].imshow(img2[0].numpy(), cmap='gray')
    axes[0, -1].set_title(f'Target\n{dataset.shape_types[meta2["shape_type"]]}')
    axes[1, -1].imshow(img2[0].numpy(), cmap='gray')
    
    # Generate and plot interpolations
    for i, alpha in enumerate(alphas):
        pixel_img = dataset.interpolate(idx1, idx2, alpha, interpolation_type="pixel")
        param_img = dataset.interpolate(idx1, idx2, alpha, interpolation_type="parameter")
        
        # Plot pixel-space interpolation
        axes[0, i+1].imshow(pixel_img[0].numpy(), cmap='gray')
        axes[0, i+1].set_title(f'Pixel-space\nα={alpha:.1f}')
        
        # Plot parameter-space interpolation
        axes[1, i+1].imshow(param_img[0].numpy(), cmap='gray')
        axes[1, i+1].set_title(f'Parameter-space\nα={alpha:.1f}')
    
    # Remove axes ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Create dataset
    dataset = ShapeDataset()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Define alpha values for interpolation
    alphas = [0.2, 0.4, 0.6, 0.8]
    
    # Test different shape combinations
    shape_pairs = [
        (0, 3),  # rectangle to rectangle
        (1, 4),  # triangle to triangle
        (2, 5)   # circle to circle   
    ]
    
    for pair_idx, (idx1, idx2) in enumerate(shape_pairs):
        # Create interpolation grid
        fig = plot_interpolation_grid(dataset, idx1, idx2, alphas)
        
        # Save the figure
        fig.savefig(f'interpolation_grid_{pair_idx}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Generated interpolation grid for shape pair {pair_idx}")


if __name__ == "__main__":
    main()
