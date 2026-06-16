from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import logging

from .config import add_common_args, apply_cli_overrides, load_config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: dict) -> None:
    import json

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dataset_summary(x_path: Path, y_path: Path) -> dict:
    x = np.load(str(x_path), mmap_mode="r")
    y = np.load(str(y_path), mmap_mode="r").reshape(-1)
    labels, counts = np.unique(y.astype(int), return_counts=True)
    return {
        "x_path": str(x_path),
        "y_path": str(y_path),
        "x_shape": tuple(int(v) for v in x.shape),
        "x_dtype": str(x.dtype),
        "y_shape": tuple(int(v) for v in y.shape),
        "y_dtype": str(y.dtype),
        "label_counts": {int(k): int(v) for k, v in zip(labels, counts)},
    }


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    data_root = Path(cfg["data"]["root"])
    report = {}
    for split in ("train", "val", "test"):
        x_path = data_root / cfg["data"][f"{split}_x"]
        y_path = data_root / cfg["data"][f"{split}_y"]
        info = dataset_summary(x_path, y_path)
        y = np.load(str(y_path), mmap_mode="r").reshape(-1).astype(int)
        info["positive_ratio"] = float(np.mean(y == int(cfg["data"].get("positive_label", 1))))
        report[split] = info
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "_audit")
    save_json(out_dir / "data_report.json", report)
    for split, info in report.items():
        logging.info(f"{split}: shape={info['x_shape']} labels={info['label_counts']} pos_ratio={info['positive_ratio']:.4f}")
    logging.info(f"Saved report to {out_dir / 'data_report.json'}")


if __name__ == "__main__":
    main()
