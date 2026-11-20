# Data

Place the spiral and wave handwriting images here. The dataset is NOT included in this repository.

Expected layout (one folder per task, one sub-folder per class):

```
data/
├── spiral/
│   ├── healthy/      *.png / *.jpg
│   └── parkinson/    *.png / *.jpg
└── wave/
    ├── healthy/
    └── parkinson/
```

Sub-folders such as `train/` and `test/` above the class folders are fine. All images found
are merged and re-split by the training script (stratified k-fold).

Add here: dataset name, source link, licence, number of images per class.
