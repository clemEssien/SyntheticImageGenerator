import os
import sys
import logging
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetMapper

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom utility for registering synthetic dataset
from utils.register_synthetic import register_synthetic_dataset

# Register the synthetic dataset
register_synthetic_dataset()

# Setup logging
logger = setup_logger(output="./output/training.log")
logger.info("Starting training...")

class EarlyStoppingHook(HookBase):
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.counter = 0
        self.early_stop_triggered = False
        
    def after_step(self):
        if self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            evaluator = COCOEvaluator("synthetic_val", self.trainer.cfg, False, output_dir=self.trainer.cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(self.trainer.cfg, "synthetic_val")
            eval_results = inference_on_dataset(self.trainer.model, val_loader, evaluator)
            
            val_segm_ap = eval_results["segm"]["AP"] if "segm" in eval_results else eval_results["bbox"]["AP"]
            
            if val_segm_ap is not None:
                logger.info(f"Validation metric (segm/AP): {val_segm_ap}")

                if self.best_metric - val_segm_ap > self.min_delta:
                    self.best_metric = val_segm_ap
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    logger.info("Early stopping triggered.")
                    self.early_stop_triggered = True
                    self.trainer.storage.put_scalar("early_stopping", True)
                    
            else:
                logger.warning("Segmentation AP metric not found. Check COCOEvaluator output.")

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.early_stop_triggered  = False

    def train(self):
        early_stopping_hook = EarlyStoppingHook(patience=5, min_delta=0.001)
        self.register_hooks([early_stopping_hook])
        
        # Custom training loop with early stopping
        super().train()
        if early_stopping_hook.early_stop_triggered:
            print("Training stopped early due to lack of improvement in validation metric.")


def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("synthetic_train",)
    cfg.DATASETS.TEST = ("synthetic_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.TEST.EVAL_PERIOD = 400
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.INPUT.MASK_FORMAT = "bitmask"
    mapper = DatasetMapper(cfg, is_train=True)
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1500, 1800)  # Adjust learning rate at these iterations
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    if not trainer.early_stop:
        evaluator = COCOEvaluator("synthetic_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "synthetic_val")
        logger.info("Starting final model evaluation on validation dataset...")
        evaluation_results = inference_on_dataset(trainer.model, val_loader, evaluator)
        logger.info(f"Final evaluation results: {evaluation_results}")
    
        with open(os.path.join(cfg.OUTPUT_DIR, "final_evaluation_results.txt"), "w") as f:
            f.write(str(evaluation_results))
        print(f"Final evaluation results saved to {cfg.OUTPUT_DIR}/final_evaluation_results.txt")

if __name__ == "__main__":
    main()
