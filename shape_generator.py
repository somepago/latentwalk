import numpy as np
from PIL import Image, ImageDraw

def create_blank_image(size=(64, 64)):
    """Create a black background image."""
    return Image.new('L', size, color=0)  # 'L' mode for grayscale, 0 for black

def draw_triangle(x, y, size=10):
    """Draw a white triangle at given coordinates."""
    img = create_blank_image()
    draw = ImageDraw.Draw(img)
    
    # Calculate vertices of the triangle
    points = [
        (x, y - size),  # top
        (x - size, y + size),  # bottom left
        (x + size, y + size)   # bottom right
    ]
    draw.polygon(points, fill=255)  # 255 for white
    return img

def draw_rectangle(x, y, width=10, height=10):
    """Draw a white rectangle at given coordinates."""
    img = create_blank_image()
    draw = ImageDraw.Draw(img)
    
    # Calculate corners of the rectangle
    left = x - width//2
    top = y - height//2
    right = x + width//2
    bottom = y + height//2
    
    draw.rectangle([left, top, right, bottom], fill=255)
    return img

def draw_circle(x, y, radius=10):
    """Draw a white circle at given coordinates."""
    img = create_blank_image()
    draw = ImageDraw.Draw(img)
    
    # Calculate bounding box for the circle
    left = x - radius
    top = y - radius
    right = x + radius
    bottom = y + radius
    
    draw.ellipse([left, top, right, bottom], fill=255)
    return img

# Example usage
if __name__ == "__main__":
    # Create some example shapes
    triangle = draw_triangle(32, 32, size=15)
    triangle.save("triangle.png")
    
    rectangle = draw_rectangle(32, 32, width=20, height=15)
    rectangle.save("rectangle.png")
    
    circle = draw_circle(32, 32, radius=15)
    circle.save("circle.png")
