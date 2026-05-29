"""
生成小型合成数据集，写入项目 `datasets/` 目录，文件名与 `configs/baseline.yaml` 中一致。
用法：在项目根目录运行 `python tools/generate_dummy_dataset.py`
"""
import numpy as np
from pathlib import Path
import logging

root = Path("datasets")
root.mkdir(exist_ok=True)

names = {
    "train": ("x_train_4201.npy", "y_train_4201.npy", 200),
    "val": ("x_val_4201.npy", "y_val_4201.npy", 50),
    "test": ("x_test_4201.npy", "y_test_4201.npy", 50),
}

H = 224
W = 224
C = 4
SCALE_MAX = 2047

for split, (xname, yname, n) in names.items():
    x = (np.random.randint(0, SCALE_MAX + 1, size=(n, H, W, C), dtype=np.uint16))
    # Balanced labels 0/1
    y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.uint8)
    # shuffle
    idx = np.random.permutation(n)
    x = x[idx]
    y = y[idx]
    np.save(root / xname, x)
    np.save(root / yname, y)

# create model_data/cls_classes.txt referenced by config (optional)
md = Path("model_data")
md.mkdir(exist_ok=True)
with (md / "cls_classes.txt").open("w", encoding="utf-8") as f:
    f.write("background\nforeground\n")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.info("Dummy datasets written to 'datasets/'")
