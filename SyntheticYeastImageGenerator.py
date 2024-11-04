import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
import random

class SyntheticYeastImageGenerator:
    def __init__(self, width=2048, height=2048):
        """
        Initialize the generator with specified image dimensions.
        
        Args:
            width (int): Width of the output image.
            height (int): Height of the output image.
        """
        self.width = width
        self.height = height
        self.image = np.zeros((height, width), dtype=np.uint16)
        self.label_image = np.zeros((height, width), dtype=np.uint8)
        
    def generate_cells(self, num_cells=50, cell_size_range=(10, 50), 
                       fluorescence_level=1000, noise_level=100):
        """
        Generate synthetic yeast cells with fluorescence in the image and a labeled mask.
        
        Args:
            num_cells (int): Number of yeast cells to generate.
            cell_size_range (tuple): Minimum and maximum cell diameter in pixels.
            fluorescence_level (int): Mean fluorescence intensity of cells.
            noise_level (int): Standard deviation of background noise.
        """
        # Initialize images
        self.image.fill(0)
        self.label_image.fill(0)
        
        # Create each cell
        for cell_id in range(1, num_cells + 1):
            self._add_cell(cell_id, cell_size_range, fluorescence_level)
        
        # Apply background noise
        self.image += np.random.normal(0, noise_level, self.image.shape).astype(np.uint16)
        self.image = np.clip(self.image, 0, np.iinfo(np.uint16).max)  # Ensure valid uint16 range

    def _add_cell(self, cell_id, cell_size_range, fluorescence_level):
        """
        Add a single cell to the synthetic image and label image.
        
        Args:
            cell_id (int): Unique identifier for the cell (used in label image).
            cell_size_range (tuple): Minimum and maximum cell diameter.
            fluorescence_level (int): Intensity of cell fluorescence.
        """
        # Random cell position and size
        x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
        cell_radius = random.randint(cell_size_range[0] // 2, cell_size_range[1] // 2)

        # Generate a circular cell shape with a Gaussian blur for realism
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= cell_radius**2
        cell_intensity = fluorescence_level + np.random.randint(-50, 50)
        
        # Update images
        self.image[mask] = np.clip(self.image[mask] + cell_intensity, 0, np.iinfo(np.uint16).max)
        self.label_image[mask] = cell_id

    def get_images(self):
        """
        Retrieve the generated fluorescence image and label image.
        
        Returns:
            tuple: (fluorescence_image, label_image)
        """
        return self.image, self.label_image

    def save_images(self, fluorescence_path="fluorescence_image.png", label_path="label_image.png"):
        """
        Save the generated images to files.
        
        Args:
            fluorescence_path (str): Path to save the fluorescence image.
            label_path (str): Path to save the label image.
        """
        plt.imsave(fluorescence_path, self.image, cmap='hot', format='png')
        plt.imsave(label_path, self.label_image, cmap='gray', format='png')

    def display_images(self):
        """
        Display the generated fluorescence and label images side by side.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image, cmap='hot')
        ax[0].set_title("Fluorescence Image")
        ax[1].imshow(self.label_image, cmap='gray')
        ax[1].set_title("Labeled Image")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters for the synthetic image generation
    generator = SyntheticYeastImageGenerator(width=512, height=512)
    generator.generate_cells(num_cells=15, cell_size_range=(20, 40), fluorescence_level=2000, noise_level=50)
    generator.display_images()
    generator.save_images("fluorescence_image.png", "label_image.png")
