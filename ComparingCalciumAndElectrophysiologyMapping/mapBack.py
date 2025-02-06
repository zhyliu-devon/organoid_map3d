from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1_path = 'data/button.png'
image2_path = 'data/image3.png'

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Find the smaller size
smallest_size = min(image1.size[0], image1.size[1], image2.size[0], image2.size[1])

# Resize both images to the smallest size using the updated resampling method
image1_resized = image1.resize((smallest_size, smallest_size), Image.Resampling.LANCZOS)
image2_resized = image2.resize((smallest_size, smallest_size), Image.Resampling.LANCZOS)

# Convert images to numpy matrices
image1_matrix = np.array(image1_resized)
image2_matrix = np.array(image2_resized)

# Save the processed matrices
np.save('data/image1_matrix.npy', image1_matrix)
np.save('data/image2_matrix.npy', image2_matrix)

# Plot the resized images
plt.figure(figsize=(10, 5))

# Plot the first image
plt.subplot(1, 2, 1)
plt.title("Resized Image 1")
plt.imshow(image1_resized)
plt.axis('off')

# Plot the second image
plt.subplot(1, 2, 2)
plt.title("Resized Image 2")
plt.imshow(image2_resized)
plt.axis('off')

plt.tight_layout()
plt.show()

