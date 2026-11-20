"""Evaluation helpers: loss, accuracy, precision, recall, F1, confusion matrix, ROC curve."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from .dataset import HandwritingDataset, list_samples
from .model import ResNet50CapsNet, margin_loss

CLASS_NAMES = ["Healthy", "Parkinson's"]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred, y_score = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lengths = model(x)
        total_loss += margin_loss(lengths, y).item() * x.size(0)
        n += x.size(0)
        y_true += y.cpu().tolist()
        y_pred += lengths.argmax(1).cpu().tolist()
        y_score += lengths[:, 1].cpu().tolist()
    return total_loss / n, np.array(y_true), np.array(y_pred), np.array(y_score)


def compute_metrics(y_true, y_pred, y_score, loss):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")
    return {"loss": float(loss), "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1), "roc_auc": float(auc)}


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a folder of images.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels = list_samples(args.data_dir)
    loader = DataLoader(HandwritingDataset(paths, labels, args.img_size, train=False),
                        batch_size=args.batch_size)
    model = ResNet50CapsNet(pretrained=False, img_size=args.img_size).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    loss, y_true, y_pred, y_score = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_score, loss)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(y_true, y_pred, out / "confusion_matrix.png")
    save_roc(y_true, y_score, out / "roc_curve.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
