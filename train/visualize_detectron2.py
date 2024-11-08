# train/visualize_detectron2.py
import cv2
import random

from detectron2 import model_zoo
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

import os
import sys

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.register_synthetic import register_synthetic_dataset

# Register datasets
register_synthetic_dataset()


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TEST = ("synthetic_train",)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("synthetic_train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("synthetic_train"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
cv2.destroyAllWindows()
