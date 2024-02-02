import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Function to load images
def load_images(base_path):
    image_data = []
    for i in range(2, 4):
        for j in range(1, 6):
            # Construct the full file path
            file_path = os.path.join(base_path, f'F1R{j}Ch{i}.png')
            
            # Check if the file exists
            if os.path.isfile(file_path):
                # Load the image and convert it to grayscale
                img = Image.open(file_path).convert('L')
                image_data.append(np.array(img, dtype=np.uint16))
    
    print(f'Loaded {len(image_data)} images')
    return image_data

# Provide the base path where the images are stored
base_path = '/media/mldadmin/home/s122mdg39_05/Projects_mrna/data/IM41340_0124'

# Create subplots for the images
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

# Load images
images = load_images(base_path)

# Loop through images and plot histograms
for i in range(3):
    for j in range(5):
        # Extract image data
        image_data = images[i * 5 + j]

        # Calculate histogram
        hist, bins = np.histogram(image_data.flatten(), bins=256, range=[0, 256])

        # Plot histogram
        axes[i, j].bar(bins[:-1], hist, width=1, color='gray')

        # Set axis labels and title
        axes[i, j].set_xlabel('Pixel Intensity')
        axes[i, j].set_ylabel('Frequency')
        axes[i, j].set_title(f'F{i + 1}R{j + 2}C{i + 1}')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.savefig('histogram.png')
