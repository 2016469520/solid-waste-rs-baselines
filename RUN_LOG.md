# Run Log

## 2026-05-29

Repository name chosen: `solid-waste-rs-baselines`

Local validation:

```text
python -m compileall solid_waste_baselines\src
python -m solid_waste_baselines.src.swbaseline.audit_data --config solid_waste_baselines\configs\baseline.yaml
python solid_waste_baselines\tools\convert_big_npy_to_dataset_v2.py --config solid_waste_baselines\configs\baseline.yaml --output-dir solid_waste_baselines\data_v2 --preview-mode rgb
```

Data audit result:

```text
train: shape=(2756, 224, 224, 4), labels={0: 1119, 1: 1637}, pos_ratio=0.5940
val:   shape=(716, 224, 224, 4), labels={0: 243, 1: 473}, pos_ratio=0.6606
test:  shape=(729, 224, 224, 4), labels={0: 250, 1: 479}, pos_ratio=0.6571
```

Inspectable dataset conversion result:

```text
total samples: 4201
single-sample npy files: 4201
preview png files: 4201
samples.csv: generated
summary.json: generated
```

Notes:

- Generated data, model checkpoints and prediction artifacts are intentionally ignored by Git.
- Use `tools/convert_big_npy_to_dataset_v2.py` to regenerate inspectable local data.

