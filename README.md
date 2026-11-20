# Hybrid ResNet50–CapsNet for Early Parkinson's Disease Detection from Handwriting

PyTorch implementation of a hybrid deep learning framework that classifies **spiral** and **wave**
handwriting images as *Healthy* or *Parkinson's disease (PD)*. ResNet50 extracts deep features and a
Capsule Network (dynamic routing) performs the classification.

## Method

```
Handwriting image
  -> Gaussian smoothing + adaptive thresholding        (src/preprocess.py)
  -> Augmentation: rotate, scale, shift, shear         (src/dataset.py)
  -> ResNet50 feature extractor                        (src/model.py)
  -> Primary capsules -> dynamic routing (3 iterations)
  -> Class capsules; capsule length = class score
  -> Margin loss, stratified 5-fold cross-validation   (src/train.py)
```

## Results reported in the paper

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN (basic) | 95.8% | 95.2% | 95.0% | 95.1% |
| ResNet50 | 97.6% | 97.2% | 97.0% | 97.1% |
| EfficientNet | 98.3% | 98.0% | 97.8% | 97.9% |
| CapsNet | 98.9% | 98.6% | 98.4% | 98.5% |
| **ResNet50 + CapsNet (proposed)** | **99.21%** | 99.47% | 99.30% | 99.38% |

5-fold cross-validation: mean accuracy **97.67% ± 1.05%**.

### Results from running this repository

| Dataset | Folds | Mean accuracy | Std | ROC-AUC |
|---|---|---|---|---|
| Spiral | 5 | _fill after running_ | | |
| Wave | 5 | _fill after running_ | | |

Add `results/confusion_matrix_fold1.png` and `results/roc_fold1.png` here once generated.

## Setup

```bash
git clone https://github.com/<your-username>/parkinsons-resnet50-capsnet.git
cd parkinsons-resnet50-capsnet
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the images in `data/spiral/{healthy,parkinson}` and `data/wave/{healthy,parkinson}`
(see `data/README.md`).

## Train

```bash
python -m src.train --data_dir data/spiral --out_dir results/spiral --epochs 10 --folds 5
python -m src.train --data_dir data/wave   --out_dir results/wave   --epochs 10 --folds 5
```

Useful flags: `--folds 1` (single 80/20 split), `--freeze_backbone`, `--no_pretrained`,
`--batch_size`, `--lr`, `--seed`.

## Evaluate a saved model

```bash
python -m src.evaluate --data_dir data/spiral --ckpt results/spiral/best_fold1.pth --out_dir results/spiral/eval
```

Outputs `metrics.json`, `confusion_matrix.png` and `roc_curve.png`.

## Project structure

```
src/preprocess.py   Gaussian filter + adaptive threshold
src/dataset.py      data loading and augmentation
src/model.py        ResNet50 + CapsNet, squash, margin loss
src/train.py        k-fold training
src/evaluate.py     metrics, confusion matrix, ROC
```

## Notes

- Capsule dimensions, learning rate and augmentation ranges are documented defaults in the code;
  adjust them to match your experiments.
- Research use only. This is not a medical diagnostic tool.

```
