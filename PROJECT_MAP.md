# Project Map

This folder is a clean rework of the solid-waste classification task.

## What Each Folder Means

```text
solid_waste_baselines/
  configs/                 User-facing experiment settings.
  data_v2/                 Inspectable dataset generated from old big npy files.
  outputs/                 Audit reports and old default output folder.
  runs/                    Recommended place for future model checkpoints/results.
  scripts/                 One-command runners.
  tools/                   Data conversion and maintenance tools.
  src/swbaseline/          Reusable Python package used by training/evaluation/inference.
```

## Recommended Workflow

1. Convert old big `.npy` arrays into inspectable `data_v2`.
2. Open `data_v2/samples.csv` and `data_v2/previews/...` to inspect label quality.
3. Train several baselines with the same config.
4. Pick thresholds from validation/test threshold sweep tables.
5. Run large-area inference and collect false positives for hard-negative mining.

## Important Files

- `configs/baseline.yaml`: the main experiment configuration.
- `data_v2/samples.csv`: one row per sample, with label, split, paths, shape and source.
- `src/swbaseline/train.py`: training entry.
- `src/swbaseline/evaluate.py`: test/evaluation entry.
- `src/swbaseline/infer_patches.py`: large-area patch inference entry.
- `tools/convert_big_npy_to_dataset_v2.py`: converts old big `.npy` arrays to inspectable files.

## Why Keep `src/`

`src/swbaseline` is the code package. It looks abstract, but the idea is simple:

- `datasets.py`: how to read patch data.
- `models.py`: which baseline model to build.
- `metrics.py`: how to calculate F1, recall, Kappa and threshold sweeps.
- `train.py`: training loop.
- `evaluate.py`: evaluation and prediction export.
- `infer_patches.py`: large-area prediction export.

The user-facing things are mostly `configs/`, `data_v2/`, `runs/`, and `scripts/`.

