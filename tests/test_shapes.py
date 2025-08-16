from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.shape_dataset import ShapeDataset

def main():
    # Create dataset
    dataset = ShapeDataset(num_samples=100, image_size=224, min_size=32, max_size=128)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Get a batch of images
    images, metadata = next(iter(dataloader))
    # Plot the images
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()
    
    for i in range(4):
        axes[i].imshow(images[i, 0], cmap='gray')
        shape_type = metadata['shape_type'][i]
        pos = metadata['position'][i]
        size = metadata['size'][i]
        axes[i].set_title(f"{shape_type}\npos:{pos}, size:{size}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_shapes.png')
    print("Generated sample_shapes.png with 4 random shapes")

if __name__ == "__main__":
    main()
