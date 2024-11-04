# Synthetic Yeast Cell Image Generator

## Project Overview
This project generates synthetic fluorescence microscopy images of yeast cells for training cell segmentation algorithms. The output includes a fluorescence image and a corresponding labeled image that uniquely identifies each cell.

## Requirements
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`

To install the required packages, run:
```bash
pip install numpy scipy matplotlib

## Usage
The main file, synthetic_image_generator.py, contains the SyntheticYeastImageGenerator class. Below is an example script demonstrating how to use the generator.

from synthetic_image_generator import SyntheticYeastImageGenerator

# Initialize the generator with desired image dimensions
generator = SyntheticYeastImageGenerator(width=512, height=512)

# Generate images with custom parameters
generator.generate_cells(num_cells=15, cell_size_range=(20, 40), fluorescence_level=2000, noise_level=50)

# Display the generated images
generator.display_images()

# Save images
generator.save_images("fluorescence_image.png", "label_image.png")
