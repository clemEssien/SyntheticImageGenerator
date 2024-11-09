# Synthetic Yeast Cell Image Generator

## Project Overview
This project generates synthetic fluorescence microscopy images of yeast cells for training cell segmentation algorithms. It includes tools to create synthetic images, convert them to COCO format for object detection models, and a model training setup using Detectron2.

## Prerequisites
- Anaconda / Miniconda

## Environment Setup
First, create and activate a new conda environment:
```bash
conda create -n env3 python=3.8
```
From the anaconda prompt, run: 
```bash 
conda activate env3
```
### Install Dependencies
Install PyTorch, torchvision, torchaudio, and cudatoolkit: <br>
``` conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch ```
``` pip install cython ```

<!-- pip install opencv-python -->

## Setting up detectron2
From the project root directory, run the following commands:
```bash 
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 
pip install -e .
```
```bash
conda install -c conda-forge opencv cython pycocotools streamlit
``` 

## Project Structure

```bash <pre>
SyntheticImageGenerator/
├── app/
│   ├── ui.py                    # Streamlit UI for generating synthetic images
│   └── image_gen.py             # SyntheticImageGenerator class implementation
├── data/
│   ├── synthetic_dataset/
│   │   ├── train/               # Training images and labels
│   │   ├── val/                 # Validation images and labels
│   │   ├── train_annotations.json    # COCO-format training annotations file
│   │   └── val_annotations.json      # COCO-format validation annotations file
├── models/                      # Folder for saving trained models
│   └── mask_rcnn_synthetic.pth  # Trained model file
├── output/                      # Output directory for Detectron2
│   ├── model_final.pth          # Trained model weights
│   └── metrics.json             # Training and evaluation metrics
├── train/
│   ├── train_detectron2.py      # Training script for Detectron2
│   ├── visualize_detectron2.py  # Visualization script for model predictions
│   └── plot_training_curves.py  # Script for plotting training loss curves
├── utils/
│   ├── coco_format.py           # Script to convert data to COCO format
│   └── register_synthetic.py    # Script to register dataset with Detectron2
├── requirements.txt             # List of project dependencies
└── README.md                    # Project documentation

  ```
        
## Usage
Navigate back to the root directory and following the instructions below
1. To generate images from command line, run the command: 
```bash
python app/image_gen.py
``` 
2. To generate images using the UI and prepare them for model training run the command:
```bash 
   python streamlit run app/ui.py 
```

Set parameters, when Generate Batch button is clicked, the images are generated in ``` data/synthetic_dataset``` directory by default, then the images are automatically split into training and validation in the ratio 80:20
and are saved in ```data/synthetic_dataset/train and data/synthetic_dataset/val ``` respectively.

3. To convert to COCO Format: Convert the generated images to COCO format for training run the following command: 
```bash 
python utils/coco_format.py --dataset_dir data/synthetic_dataset/train --output_file data/synthetic_dataset/train_annotations.json 
```
```bash 
python utils/coco_format.py --dataset_dir data/synthetic_dataset/val --output_file data/synthetic_dataset/val_annotations.json
``` 
- To inspect the annotations:
```bash 
python inspect_coco_annotations.py 
```

4. Train Model: Use the training script to train a model with Detectron2.
```bash
python train/train_detectron2.py
```

5. Plot Training Curves: Visualize training loss and other metrics.

```bash
python train/plot_training_curves.py
```

# Results
The final results will be saved in 
```bash 
output/final_evaluation_results.txt 
```