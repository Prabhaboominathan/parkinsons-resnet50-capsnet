"""Train and evaluate ResNet50-CapsNet with stratified k-fold cross-validation.

Example:
    python -m src.train --data_dir data/spiral --out_dir results/spiral --epochs 10 --folds 5
Use --folds 1 for a single 80/20 stratified split.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from .dataset import HandwritingDataset, list_samples
from .evaluate import compute_metrics, predict, save_confusion_matrix, save_roc
from .model import ResNet50CapsNet, margin_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(fold, tr_idx, va_idx, paths, labels, args, device, out_dir):
    tr = HandwritingDataset([paths[i] for i in tr_idx], [labels[i] for i in tr_idx], args.img_size, train=True)
    va = HandwritingDataset([paths[i] for i in va_idx], [labels[i] for i in va_idx], args.img_size, train=False)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    va_loader = DataLoader(va, batch_size=args.batch_size, num_workers=args.workers)

    model = ResNet50CapsNet(pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone,
                            img_size=args.img_size).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_acc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = margin_loss(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        val_loss, y_true, y_pred, y_score = predict(model, va_loader, device)
        m = compute_metrics(y_true, y_pred, y_score, val_loss)
        history.append({"epoch": epoch, "train_loss": running / seen, **m})
        print(f"[fold {fold}] epoch {epoch:02d}  train_loss {running / seen:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {m['accuracy']:.4f}")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            torch.save(model.state_dict(), out_dir / f"best_fold{fold}.pth")

    final = history[-1]                      # metrics after the last epoch
    save_confusion_matrix(y_true, y_pred, out_dir / f"confusion_matrix_fold{fold}.png")
    save_roc(y_true, y_score, out_dir / f"roc_fold{fold}.png")
    return final, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="results/run")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true", help="random-init ResNet50")
    ap.add_argument("--freeze_backbone", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = list_samples(args.data_dir)
    labels_arr = np.array(labels)
    print(f"{len(paths)} images | healthy={int((labels_arr == 0).sum())} "
          f"parkinson={int((labels_arr == 1).sum())} | device={device}")

    if args.folds > 1:
        splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                      .split(np.zeros(len(labels)), labels_arr))
    else:
        idx = np.arange(len(labels))
        tr, va = train_test_split(idx, test_size=0.2, stratify=labels_arr, random_state=args.seed)
        splits = [(tr, va)]

    fold_results, histories = [], []
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        final, hist = run_fold(fold, tr_idx, va_idx, paths, labels, args, device, out_dir)
        fold_results.append(final)
        histories.append(hist)

    accs = np.array([r["accuracy"] for r in fold_results])
    summary = {
        "folds": len(fold_results),
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "mean_loss": float(np.mean([r["loss"] for r in fold_results])),
        "per_fold": fold_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(histories, indent=2))
    print(f"Mean accuracy {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
