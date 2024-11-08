# inspect_coco_annotations.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

def load_image(img_path):
    """Load an image from the file path."""
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def draw_annotations(img, annotations, coco):
    """Draw bounding boxes and masks on an image."""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    for ann in annotations:
        # Draw bounding box
        bbox = ann['bbox']
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Draw segmentation mask (if available)
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # Polygon format
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    ax.plot(poly[:, 0], poly[:, 1], color='yellow', linewidth=2)
            elif isinstance(ann['segmentation'], dict):
                # RLE format
                rle = ann['segmentation']
                if isinstance(rle['counts'], list):
                    # Uncompressed RLE
                    mask = mask_util.decode(rle)
                else:
                    # Compressed RLE
                    mask = mask_util.decode(rle)

                # Apply mask with transparency
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='cool', alpha=0.5)

    plt.axis('off')
    plt.show()

def inspect_annotations(dataset_dir, annotations_file, num_images=5):
    """Load COCO annotations and visualize them."""
    coco = COCO(annotations_file)
    img_ids = coco.getImgIds()

    # Display a few images
    for img_id in img_ids[:num_images]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_dir, img_info['file_name'])
        img = load_image(img_path)

        # Get all annotations for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Draw annotations on the image
        print(f"Displaying annotations for image: {img_info['file_name']}")
        draw_annotations(img, anns, coco)

if __name__ == "__main__":
    dataset_dir = "data/synthetic_dataset/val"  # Directory containing validation images
    annotations_file = "data/synthetic_dataset/val_annotations.json"  # Path to COCO annotations file
    inspect_annotations(dataset_dir, annotations_file, num_images=5)
