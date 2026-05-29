from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image
import logging


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stretch_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def make_preview(image: np.ndarray, mode: str = "rgb") -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={image.shape}")
    channels = image.shape[2]
    if channels == 1:
        preview = stretch_to_uint8(image[:, :, 0])
        return Image.fromarray(preview, mode="L")
    if mode == "false_color" and channels >= 4:
        use = image[:, :, [3, 2, 1]]
    else:
        use = image[:, :, : min(3, channels)]
        if use.shape[2] == 2:
            use = np.concatenate([use, use[:, :, :1]], axis=2)
    preview = np.stack([stretch_to_uint8(use[:, :, i]) for i in range(use.shape[2])], axis=2)
    return Image.fromarray(preview, mode="RGB")


def iter_splits(cfg: Dict) -> Iterable[Tuple[str, Path, Path]]:
    root = Path(cfg["data"]["root"])
    for split in ("train", "val", "test"):
        yield split, root / cfg["data"][f"{split}_x"], root / cfg["data"][f"{split}_y"]


def write_samples_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "sample_id",
        "split",
        "label",
        "label_name",
        "image_path",
        "preview_path",
        "source_x",
        "source_y",
        "source_index",
        "height",
        "width",
        "channels",
        "dtype",
        "min",
        "max",
        "mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def convert(cfg: Dict, output_dir: Path, preview_mode: str, limit: int | None = None) -> Dict:
    classes_path = Path(cfg["data"].get("classes", "model_data/cls_classes.txt"))
    if classes_path.exists():
        class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        class_names = ["nonsw", "sw"]

    images_root = ensure_dir(output_dir / "images")
    previews_root = ensure_dir(output_dir / "previews")
    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for split, x_path, y_path in iter_splits(cfg):
        x_arr = np.load(str(x_path), mmap_mode="r")
        y_arr = np.load(str(y_path), mmap_mode="r").reshape(-1).astype(int)
        if len(x_arr) != len(y_arr):
            raise ValueError(f"Length mismatch for {split}: {x_path}={len(x_arr)}, {y_path}={len(y_arr)}")

        split_image_dir = ensure_dir(images_root / split)
        split_preview_dir = ensure_dir(previews_root / split)
        count = min(len(y_arr), limit) if limit is not None else len(y_arr)
        label_counts: Dict[int, int] = {}

        for idx in range(count):
            image = np.asarray(x_arr[idx])
            label = int(y_arr[idx])
            label_counts[label] = label_counts.get(label, 0) + 1
            sample_id = f"{split}_{idx:06d}"
            image_rel = Path("images") / split / f"{sample_id}.npy"
            preview_rel = Path("previews") / split / f"{sample_id}.png"
            np.save(output_dir / image_rel, image)
            make_preview(image, mode=preview_mode).save(output_dir / preview_rel)

            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "label": label,
                    "label_name": class_names[label] if 0 <= label < len(class_names) else str(label),
                    "image_path": str(image_rel).replace("\\", "/"),
                    "preview_path": str(preview_rel).replace("\\", "/"),
                    "source_x": str(x_path).replace("\\", "/"),
                    "source_y": str(y_path).replace("\\", "/"),
                    "source_index": idx,
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                    "channels": int(image.shape[2]) if image.ndim == 3 else 1,
                    "dtype": str(image.dtype),
                    "min": float(np.min(image)),
                    "max": float(np.max(image)),
                    "mean": float(np.mean(image)),
                }
            )

        summary[split] = {
            "source_x": str(x_path),
            "source_y": str(y_path),
            "source_shape": tuple(int(v) for v in x_arr.shape),
            "converted": count,
            "label_counts": label_counts,
        }

    write_samples_csv(output_dir / "samples.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"total": len(rows), "splits": summary}, f, ensure_ascii=False, indent=2)
    return {"total": len(rows), "splits": summary, "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert big x/y npy arrays into inspectable dataset_v2.")
    parser.add_argument("--config", default="solid_waste_baselines/configs/baseline.yaml")
    parser.add_argument("--output-dir", default="solid_waste_baselines/data_v2")
    parser.add_argument("--preview-mode", choices=["rgb", "false_color"], default="rgb")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split sample limit for a dry run.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    result = convert(cfg, Path(args.output_dir), args.preview_mode, args.limit)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

