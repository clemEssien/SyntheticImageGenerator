import os
import random
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.draw import disk, ellipse
from typing import Tuple

class SyntheticImageGenerator:
    """
    A class to generate synthetic fluorescence microscopy images and labeled images with adjustable parameters.
    Includes methods to display, save, and generate batches of synthetic images.
    """

    def __init__(self, width: int = 128, height: int = 128, num_cells: int = 15, 
                 fluorescence_level: int = 2000, cell_min_radius: int = 15, 
                 cell_max_radius: int = 30, noise_level: float = 0.02, blur_sigma: float = 2.5, 
                 overlap_percentage: float = 0.15, background_intensity: int = 50):
        self.width = width
        self.height = height
        self.num_cells = num_cells
        self.fluorescence_level = fluorescence_level
        self.cell_min_radius = cell_min_radius
        self.cell_max_radius = cell_max_radius
        self.noise_level = noise_level
        self.blur_sigma = blur_sigma
        self.overlap_percentage = overlap_percentage
        self.background_intensity = background_intensity
        self.fluorescence_image = None
        self.labeled_image = None

    def generate_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic fluorescence image and a labeled image with controlled overlap."""
        
        # Initialize background with low-level intensity
        fluorescence_image = np.full((self.height, self.width), self.background_intensity, dtype=np.uint16)
        labeled_image = np.zeros((self.height, self.width), dtype=np.uint8)

        for cell_id in range(1, self.num_cells + 1):
            # Generate cell radius and position with overlap handling
            radius = np.random.randint(self.cell_min_radius, self.cell_max_radius)
            overlap_margin = int(radius * (1 - self.overlap_percentage))
            cx, cy = np.random.randint(overlap_margin, self.width - overlap_margin), \
                     np.random.randint(overlap_margin, self.height - overlap_margin)

            # Randomly choose circular or elliptical shape
            if np.random.rand() > 0.5:
                rr, cc = disk((cy, cx), radius, shape=fluorescence_image.shape)
            else:
                minor_axis = radius
                major_axis = radius + np.random.randint(1, 5)
                orientation = np.random.rand() * np.pi
                rr, cc = ellipse(cy, cx, minor_axis, major_axis, rotation=orientation, shape=fluorescence_image.shape)

            fluorescence_intensity = np.random.randint(self.fluorescence_level // 2, self.fluorescence_level)
            fluorescence_image[rr, cc] = np.clip(fluorescence_image[rr, cc] + fluorescence_intensity, 0, 65535)
            labeled_image[rr, cc] = cell_id

        # Apply Gaussian blur and noise
        fluorescence_image = cv2.GaussianBlur(fluorescence_image.astype(np.float32), (0, 0), self.blur_sigma)
        noise = np.random.normal(0, self.fluorescence_level * self.noise_level, fluorescence_image.shape).astype(np.float32)
        self.fluorescence_image = np.clip(fluorescence_image + noise, 0, 65535).astype(np.uint16)
        self.labeled_image = labeled_image
        return self.fluorescence_image, self.labeled_image

    def display_images(self):
        """Display the synthetic fluorescence image and labeled image side-by-side."""
        if self.fluorescence_image is None or self.labeled_image is None:
            raise ValueError("Images not generated yet. Call generate_image() first.")
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        c1 = axs[0].imshow(self.fluorescence_image, cmap='inferno')
        fig.colorbar(c1, ax=axs[0])
        axs[0].set_title("Fluorescence Image")
        
        c2 = axs[1].imshow(self.labeled_image, cmap='viridis')
        fig.colorbar(c2, ax=axs[1])
        axs[1].set_title("Labeled Image")
        plt.show()

    def save_images(self, fluorescence_path: str, labeled_path: str):
        """Save the generated images to specified file paths."""
        if self.fluorescence_image is None or self.labeled_image is None:
            raise ValueError("Images not generated yet. Call generate_image() first.")
        
        cv2.imwrite(fluorescence_path, self.fluorescence_image)
        cv2.imwrite(labeled_path, self.labeled_image)
        print(f"Images saved as {fluorescence_path} and {labeled_path}.")

    def generate_batch(self, batch_size: int, save_dir: str):
        """
        Generate a batch of synthetic images and save them in the specified directory.
        Each image pair will be saved as <save_dir>/fluorescence_<index>.png and <save_dir>/label_<index>.png.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(batch_size):
            self.generate_image()
            fluorescence_path = os.path.join(save_dir, f"image_{i}.png")
            labeled_path = os.path.join(save_dir, f"label_{i}.png")
            self.save_images(fluorescence_path, labeled_path)
        print(f"Batch of {batch_size} image pairs saved to {save_dir}.")


    def set_parameters(self, **kwargs):
        """Dynamically set parameters for the image generation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not valid for SyntheticImageGenerator.")

    # Method to split the generated images into training and validation sets
    def split_dataset(self, save_dir: str):
        """Split the dataset into training and validation sets."""
        train_dir = os.path.join(save_dir, "train")
        val_dir = os.path.join(save_dir, "val")
        
        # Ensure directories exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Get list of generated files and shuffle them
        all_images = sorted([f for f in os.listdir(save_dir) if f.startswith("image_")])
        all_labels = sorted([f for f in os.listdir(save_dir) if f.startswith("label_")])
        paired_files = list(zip(all_images, all_labels))
        random.shuffle(paired_files)

        # Split into 80% train and 20% val
        train_split = int(len(paired_files) * 0.8)
        train_files = paired_files[:train_split]
        val_files = paired_files[train_split:]

        # Move files to train and val directories
        for image_file, label_file in train_files:
            shutil.move(os.path.join(save_dir, image_file), os.path.join(train_dir, image_file))
            shutil.move(os.path.join(save_dir, label_file), os.path.join(train_dir, label_file))

        for image_file, label_file in val_files:
            shutil.move(os.path.join(save_dir, image_file), os.path.join(val_dir, image_file))
            shutil.move(os.path.join(save_dir, label_file), os.path.join(val_dir, label_file))
        
# Example usage
if __name__ == "__main__":
    # Create an instance with default parameters
    generator = SyntheticImageGenerator()

    # Generate a single image and display it
    generator.generate_image()
    generator.display_images()

    # Save the generated images
    generator.save_images("image.png", "label_image.png")

    # Generate a batch of images and save to directory
    generator.generate_batch(batch_size=5, save_dir="synthetic_dataset")

    # Set parameters interactively (for UI integration)
    generator.set_parameters(width=256, height=256, num_cells=20)

    # Generate new images with updated parameters
    generator.generate_image()
    generator.display_images()
