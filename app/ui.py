import streamlit as st
from image_gen import SyntheticImageGenerator
import matplotlib.pyplot as plt
import time

# Initialize a SyntheticImageGenerator instance
generator = SyntheticImageGenerator()

# Main layout for parameter controls
st.title("Synthetic Image Generator")

# Input fields for adjusting the synthetic image parameters
st.header("Set Parameters for Image Generation")
width = st.number_input("Width", min_value=64, max_value=2048, value=1024, step=64)
height = st.number_input("Height", min_value=64, max_value=2048, value=1024, step=64)
num_cells = st.number_input("Number of Cells", min_value=1, max_value=500, value=generator.num_cells)
fluorescence_level = st.slider("Fluorescence Level", 1000, 5000, generator.fluorescence_level)
cell_min_radius = st.slider("Cell Min Radius", 5, 20, generator.cell_min_radius)
cell_max_radius = st.slider("Cell Max Radius", 10, 50, generator.cell_max_radius)
noise_level = st.slider("Noise Level", 0.0, 0.05, generator.noise_level, 0.01)
blur_sigma = st.slider("Blur Sigma", 0.0, 5.0, generator.blur_sigma, 0.1)
overlap_percentage = st.slider("Overlap Percentage", 0.0, 0.5, generator.overlap_percentage, 0.05)
background_intensity = st.slider("Background Intensity", 0, 500, generator.background_intensity)


# Update the generator parameters based on UI input
generator.set_parameters(
    width=width,
    height=height,
    num_cells=num_cells,
    fluorescence_level=fluorescence_level,
    cell_min_radius=cell_min_radius,
    cell_max_radius=cell_max_radius,
    noise_level=noise_level,
    blur_sigma=blur_sigma
)

# Generate image button
if st.button("Generate Image"):
    # Generate a new image based on the current parameters
    fluorescence_image, labeled_image = generator.generate_image()
    
    # Display the images
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.subheader("Fluorescence Image")
        fig, ax = plt.subplots()
        ax.imshow(fluorescence_image, cmap="inferno")
        plt.colorbar(ax.imshow(fluorescence_image, cmap="inferno"), ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Label Image")
        fig, ax = plt.subplots()
        ax.imshow(labeled_image, cmap="plasma")
        plt.colorbar(ax.imshow(labeled_image, cmap="plasma"), ax=ax)
        st.pyplot(fig)

# Batch generation options
st.header("Batch Generation")
batch_size = st.selectbox("Batch Size (Multiples of 100)", options=[100 * i for i in range(1, 11)], index=0)
save_dir = st.text_input("Save Directory", "data/synthetic_dataset")

# Generate batch button with progress bar
if st.button("Generate Batch"):
    with st.spinner("Generating batch, please wait..."):
        # Set up a progress bar
        progress_bar = st.progress(0)
        total_images = batch_size
        for i in range(total_images):
            # Generate each image and update the progress
            generator.generate_image()
            fluorescence_path = f"{save_dir}/image_{i}.png"
            labeled_path = f"{save_dir}/label_{i}.png"
            generator.save_images(fluorescence_path, labeled_path)
            progress_bar.progress((i + 1) / total_images)
        st.success(f"Batch of {batch_size} images saved to {save_dir}.")
        
    generator.split_dataset(save_dir)
# Display instructions for user
st.write("Adjust the parameters and click 'Generate Image' to preview the synthetic images. You can also save individual images or generate a batch of images.")
