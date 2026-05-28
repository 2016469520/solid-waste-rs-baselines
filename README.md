# Solid Waste Remote-Sensing Baselines

This subproject is a clean training/evaluation/inference pipeline for binary
solid-waste patch classification.

The original project already contains useful data and checkpoints, but many
scripts are experiment-specific. This folder keeps the task explicit:

- input: 4-channel remote-sensing patches, usually `(N, 224, 224, 4)` in `.npy`
- label: `0 = nonsw`, `1 = sw`
- output: trained baseline checkpoints, test metrics, threshold reports, and
  large-area patch predictions

## Directory

```text
solid_waste_baselines/
  configs/
    baseline.yaml
  data_v2/
    samples.csv
    images/
    previews/
  src/swbaseline/
    config.py
    datasets.py
    metrics.py
    models.py
    train.py
    evaluate.py
    infer_patches.py
    utils.py
  scripts/
    train_all_baselines.ps1
  tools/
    convert_big_npy_to_dataset_v2.py
  outputs/
    .gitkeep
```

## Baselines

Available model names:

- `small_cnn`
- `resnet18`
- `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `convnext_tiny`

All torchvision models are adapted to 4-channel input. If `pretrained: true`,
the first convolution is initialized from RGB weights by copying the RGB mean
into the extra channel.

## Quick Start

Run one model:

```powershell
python -m solid_waste_baselines.src.swbaseline.audit_data --config solid_waste_baselines/configs/baseline.yaml
python solid_waste_baselines/tools/convert_big_npy_to_dataset_v2.py --config solid_waste_baselines/configs/baseline.yaml --output-dir solid_waste_baselines/data_v2
python -m solid_waste_baselines.src.swbaseline.train --config solid_waste_baselines/configs/baseline.yaml --model resnet18
```

Evaluate a checkpoint:

```powershell
python -m solid_waste_baselines.src.swbaseline.evaluate --config solid_waste_baselines/configs/baseline.yaml --checkpoint solid_waste_baselines/outputs/resnet18/best.pt
```

Run several baselines:

```powershell
powershell -ExecutionPolicy Bypass -File solid_waste_baselines/scripts/train_all_baselines.ps1
```

Large-area patch inference:

```powershell
python -m solid_waste_baselines.src.swbaseline.infer_patches --config solid_waste_baselines/configs/baseline.yaml --checkpoint solid_waste_baselines/outputs/resnet18/best.pt --patch-dir /data/all_patches/7190 --output solid_waste_baselines/outputs/resnet18/7190_predictions.csv
```

## Why This Pipeline

The old large-area result can fail for several reasons: threshold mismatch,
train/test distribution shift, class imbalance, weak negative mining, or a model
that overfits local validation data. This pipeline makes those failure modes
visible by saving:

- per-epoch training and validation metrics
- test confusion matrix, precision, recall, F1, Kappa, ROC-AUC and PR-AUC
- threshold sweep table so inference thresholds can be selected by objective
- CSV predictions with filename, positive probability, and thresholded label
