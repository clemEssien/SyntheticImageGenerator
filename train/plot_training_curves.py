# plot_training_curves.py
from detectron2.utils.events import EventStorage
import matplotlib.pyplot as plt
import json
import os

def plot_losses(log_dir="output"):
    # Locate the metrics file in the output directory
    metrics_file = os.path.join(log_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        print("Metrics file not found. Ensure training has completed.")
        return

    # Load metrics
    with open(metrics_file, "r") as f:
        metrics = [json.loads(line) for line in f]

    # Extract loss and iteration data
    iterations = [x["iteration"] for x in metrics if "total_loss" in x]
    total_loss = [x["total_loss"] for x in metrics if "total_loss" in x]
    loss_cls = [x["loss_cls"] for x in metrics if "loss_cls" in x]
    loss_box_reg = [x["loss_box_reg"] for x in metrics if "loss_box_reg" in x]
    loss_mask = [x["loss_mask"] for x in metrics if "loss_mask" in x]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, total_loss, label="Total Loss")
    plt.plot(iterations, loss_cls, label="Classification Loss")
    plt.plot(iterations, loss_box_reg, label="Box Regression Loss")
    plt.plot(iterations, loss_mask, label="Mask Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_losses()
