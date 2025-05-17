import numpy as np
from PIL import Image
import os

def create_test_image(filename="test_image.jpg", size=(224, 224)):
    """Create a test image for testing the model"""
    # Create a random image
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    
    # Create a simple pattern
    for i in range(size[0]):
        for j in range(size[1]):
            if (i // 20) % 2 == (j // 20) % 2:
                img_array[i, j, :] = [200, 100, 100]  # Reddish color
            else:
                img_array[i, j, :] = [100, 200, 100]  # Greenish color
    
    # Add a circle
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 4
    
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - center_x) ** 2 + (j - center_y) ** 2 < radius ** 2:
                img_array[i, j, :] = [100, 100, 200]  # Bluish color
    
    # Convert to PIL Image and save
    img = Image.fromarray(img_array)
    img.save(filename)
    
    print(f"Test image created and saved as {filename}")
    print(f"Image size: {size}")
    
    return filename

if __name__ == "__main__":
    # Create a test image
    filename = create_test_image()
    
    # Verify the image was created
    if os.path.exists(filename):
        print(f"SUCCESS: Test image {filename} created successfully!")
    else:
        print(f"FAILURE: Failed to create test image {filename}")
