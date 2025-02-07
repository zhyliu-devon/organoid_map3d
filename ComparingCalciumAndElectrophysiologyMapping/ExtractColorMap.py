import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys
import os

def extract_colorbar(image_path):
    # Read the image
    img = imread(image_path)
    
    # Create figure for point selection
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Click two points to define color bar range\nClose window when done. Select blue first')
    
    # Get points from user clicks
    points = plt.ginput(2)
    plt.close()
    
    if len(points) != 2:
        return None
    
    # Extract coordinates
    (x1, y1), (x2, y2) = points
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    
    # Extract RGB values along vertical line
    rgb_values = []
    for y in range(min(y1, y2), max(y1, y2) + 1):
        rgb_values.append(img[y, x1])
    
    rgb_values = np.array(rgb_values)
    
    return rgb_values

def plot_colorbar(rgb_values):
    # Create figure
    plt.figure(figsize=(2, 8))
    
    # Create color bar using extracted RGB values
    height = len(rgb_values)
    color_array = np.tile(rgb_values, (20, 1, 1))  # Make it wider for visibility
    
    plt.imshow(color_array)
    plt.axis('off')
    plt.title('Extracted Color Bar')
    plt.show()

# Main execution
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    image_path = os.path.join(script_dir, "scalebar.png")
    try:
        rgb_values = extract_colorbar(image_path)
        if rgb_values is not None:
            print(f"Extracted {len(rgb_values)} RGB values")
            plot_colorbar(rgb_values)
            # Save RGB values if needed
            np.save(os.path.join(script_dir, 'colorbar_values.npy'), rgb_values)
    except Exception as e:
        print(f"Error: {e}")