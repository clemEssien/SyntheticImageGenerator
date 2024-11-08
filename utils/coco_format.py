import os
import json
from PIL import Image
import numpy as np
import argparse
from pycocotools import mask as mask_util

def create_coco_json(dataset_dir, output_file):
    images = []
    annotations = []
    annotation_id = 1  # Unique ID for each annotation

    # Get list of all image files
    all_images = sorted([f for f in os.listdir(dataset_dir) if f.startswith("image_")])
    
    for idx, image_file in enumerate(all_images):
        # Define corresponding label file
        label_file = image_file.replace("image", "label")

        # Load image and label
        image_path = os.path.join(dataset_dir, image_file)
        label_path = os.path.join(dataset_dir, label_file)

        # Open image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info to images list
        image_id = idx + 1
        images.append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # Process label to get each cell as an individual object
        label = np.array(Image.open(label_path))
        unique_labels = np.unique(label)
        
        for cell_label in unique_labels:
            if cell_label == 0:
                continue  # Skip background

            # Create mask for this cell
            cell_mask = label == cell_label

            # Convert mask to COCO-style RLE
            rle = mask_util.encode(np.asfortranarray(cell_mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")  # Convert byte string to utf-8 for JSON serialization

            # Calculate bounding box
            pos = np.where(cell_mask)
            xmin, ymin, xmax, ymax = np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # Single class, so category_id is 0
                "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
                "segmentation": rle,  # Using RLE for binary mask
                "iscrowd": 0,
                "area": int(cell_mask.sum())
            })
            annotation_id += 1

    # COCO JSON format dictionary
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "cell"}]
    }

    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    print(f"COCO format annotations saved to {output_file}")

if __name__ == "__main__":
    # Argument parser for dataset directory and output file
    parser = argparse.ArgumentParser(description="Convert synthetic dataset to COCO format")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing image and label pairs")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for COCO annotations")
    
    args = parser.parse_args()
    create_coco_json(args.dataset_dir, args.output_file)
