import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import einops


class ShapeDataset(Dataset):
    def __init__(self, num_samples=100000, image_size=64, min_size=8, max_size=20):
        """
        Args:
            num_samples (int): Number of samples in the dataset
            image_size (int): Size of the output images (square)
            min_size (int): Minimum size of shapes
            max_size (int): Maximum size of shapes
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
        
        # Define shape types
        self.shape_types = ['rectangle', 'triangle', 'circle']

        # Define shape sizes
        self.shape_sizes = np.arange(min_size, max_size + 1)
        
    def create_blank_image(self):
        """Create a black background image."""
        return Image.new('L', (self.image_size, self.image_size), color=0)
    
    def draw_shape(self, shape_type, x, y, size):
        """Draw a specific shape at given coordinates with given size."""
        img = self.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        if shape_type == 'rectangle':
            # For rectangle, size is the width, make height slightly different
            width = size
            height = int(size * random.uniform(0.75, 1.25))
            left = x - width//2
            top = y - height//2
            right = x + width//2
            bottom = y + height//2
            draw.rectangle([left, top, right, bottom], fill=255)
            
        elif shape_type == 'triangle':
            # Calculate vertices of the triangle
            points = [
                (x, y - size),  # top
                (x - size, y + size),  # bottom left
                (x + size, y + size)   # bottom right
            ]
            draw.polygon(points, fill=255)
            
        else:  # circle
            # Calculate bounding box for the circle
            left = x - size
            top = y - size
            right = x + size
            bottom = y + size
            draw.ellipse([left, top, right, bottom], fill=255)
            
        return img
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a shape image based on the index.
        The shape type is determined by idx % 3
        Position and size are determined pseudo-randomly based on idx
        """
        random.seed(idx)  # Ensure reproducibility for each index
        # Select shape type based on index
        shape_type = self.shape_types[idx % len(self.shape_types)]
        
        # Generate random position ensuring shape fits in image
        size = self.shape_sizes[idx % len(self.shape_sizes)]

        margin = size + 1  # Add margin to ensure shape doesn't touch edges
        x = random.randint(margin, self.image_size - margin)
        y = random.randint(margin, self.image_size - margin)
        
        # Draw the shape
        img = self.draw_shape(shape_type, x, y, size)
        
        # Convert to tensor
        img = torch.from_numpy(np.array(img)).float() / 255.0

        # Repeat channel dimension to fake RGB
        img = einops.repeat(img, 'h w -> c h w', c=3)

        # Create metadata dict
        meta = {
            'shape_type': idx % len(self.shape_types),
            'position': torch.tensor([x, y]),
            'size': size
        }

        return img, meta


    def interpolate(self, idx1, idx2, alpha, interpolation_type="parameter"):
        """
        Interpolate between two shapes based on their indices.
        
        Args:
            idx1 (int): Index of the first shape
            idx2 (int): Index of the second shape
            alpha (float): Interpolation factor between 0 and 1
                         0 = first shape
                         1 = second shape
            interpolation_type (str): Type of interpolation to perform ("pixel" or "parameter")
        Returns:
            interpolated_image
        """
        # Get the two shapes
        img1, meta1 = self.__getitem__(idx1)
        img2, meta2 = self.__getitem__(idx2)
        
        if interpolation_type == "pixel":
            # Pixel-space interpolation (direct blend)
            pixel_interp = img1 * (1 - alpha) + img2 * alpha
            return pixel_interp

        # Parameter-space interpolation
        interp_size = int(meta1['size'] * (1 - alpha) + meta2['size'] * alpha)
        interp_pos = meta1['position'] * (1 - alpha) + meta2['position'] * alpha
        
        # For shape type, we'll use the source shape for alpha < 0.5, target shape for alpha >= 0.5
        interp_shape_type = meta1['shape_type'] if alpha < 0.5 else meta2['shape_type']
        shape_type = self.shape_types[interp_shape_type]
        
        # Draw the interpolated shape
        param_interp_img = self.draw_shape(
            shape_type, 
            int(interp_pos[0].item()), 
            int(interp_pos[1].item()), 
            interp_size
        )
        param_interp_img = torch.from_numpy(np.array(param_interp_img)).float() / 255.0
        param_interp_img = einops.repeat(param_interp_img, 'h w -> c h w', c=3)

        return param_interp_img
