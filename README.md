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
```

## Usage
The main file, ```synthetic_image_generator.py```, contains the ```SyntheticYeastImageGenerator``` class. <br> 
Below is an example script demonstrating how to use the generator.

```
from synthetic_image_generator import SyntheticYeastImageGenerator

# Initialize the generator with desired image dimensions
generator = SyntheticYeastImageGenerator(width=512, height=512)

# Generate images with custom parameters
generator.generate_cells(num_cells=15, cell_size_range=(20, 40), fluorescence_level=2000, noise_level=50)

# Display the generated images
generator.display_images()

# Save images
generator.save_images("fluorescence_image.png", "label_image.png")

```

## Expected Output
Running the above code will generate:

1. ```fluorescence_image.png```: A synthetic fluorescence image in uint16 format.
2. ```label_image.png```: A labeled image with unique integer labels for each cell in uint8 format.

## Input Parameters
```width, height```: Dimensions of the generated images.
```num_cells```: Number of cells to generate.
```cell_size_range```: Range for cell diameters.
```fluorescence_level```: Mean fluorescence intensity of the cells.
```noise_level```: Standard deviation of background noise.

## Observations and Assumptions
- Cells are generated as circular shapes; future versions could incorporate elliptical or irregular shapes.
- Gaussian noise is added to simulate camera background noise.
- Each cell is assigned a unique integer ID up to a maximum of 255 cells per image.

## References
This project uses the following libraries:

```numpy``` for array manipulation <br>
```scipy``` for generating random distributions and Gaussian filtering <br>
```matplotlib``` for image visualization and saving <br>