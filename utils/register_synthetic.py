from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import json

def get_synthetic_dicts(dataset_dir, json_file):
    with open(json_file) as f:
        coco_data = json.load(f)

    dataset_dicts = []
    for img in coco_data["images"]:
        record = {
            "file_name": os.path.join(dataset_dir, img["file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "annotations": []
        }
        for ann in coco_data["annotations"]:
            if ann["image_id"] == img["id"]:
                ann["bbox_mode"] = 0  # This should match the bbox mode used by Detectron2
                record["annotations"].append(ann)
        dataset_dicts.append(record)
    return dataset_dicts

def register_synthetic_dataset():
    # Register training dataset
    DatasetCatalog.register("synthetic_train", lambda: get_synthetic_dicts("data/synthetic_dataset/train", "data/synthetic_dataset/train_annotations.json"))
    MetadataCatalog.get("synthetic_train").set(thing_classes=["cell"])

    # Register validation dataset
    DatasetCatalog.register("synthetic_val", lambda: get_synthetic_dicts("data/synthetic_dataset/val", "data/synthetic_dataset/val_annotations.json"))
    MetadataCatalog.get("synthetic_val").set(thing_classes=["cell"])

if __name__ == "__main__":
    try:
        register_synthetic_dataset()
        print("Synthetic datasets registered successfully.")
    except Exception as e:
        print(f"Error registering dataset: {e}")
