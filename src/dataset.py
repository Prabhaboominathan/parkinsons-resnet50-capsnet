"""Dataset utilities for spiral / wave handwriting images (Healthy = 0, Parkinson's = 1)."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocess import preprocess_image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {"healthy": 0, "normal": 0, "parkinson": 1, "parkinsons": 1, "pd": 1}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def list_samples(data_dir: str) -> Tuple[List[Path], List[int]]:
    """Collect (path, label) pairs; the label comes from the parent folder name."""
    paths, labels = [], []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.parent.name.lower() in LABEL_MAP:
            paths.append(p)
            labels.append(LABEL_MAP[p.parent.name.lower()])
    if not paths:
        raise RuntimeError(f"No images found under {data_dir}. See data/README.md for the layout.")
    return paths, labels


def build_transforms(img_size: int = 224, train: bool = True):
    """Rotate / scale / shift / shear augmentation for training (paper Eq. 4)."""
    ops = []
    if train:
        ops.append(transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                           scale=(0.9, 1.1), shear=10, fill=255))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class HandwritingDataset(Dataset):
    def __init__(self, paths, labels, img_size: int = 224, train: bool = True):
        self.paths, self.labels = list(paths), list(labels)
        self.img_size = img_size
        self.tf = build_transforms(img_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        binary = preprocess_image(self.paths[idx], self.img_size)          # (H, W) uint8
        rgb = Image.fromarray(np.stack([binary] * 3, axis=-1))            # 3 channels for ResNet50
        return self.tf(rgb), self.labels[idx]
