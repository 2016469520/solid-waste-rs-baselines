from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import add_common_args, apply_cli_overrides, load_config
from .datasets import NpyPatchDataset
from .metrics import average_precision, auc_score, best_by, binary_metrics, threshold_sweep
from .models import create_model
from .utils import ensure_dir, load_checkpoint, save_csv, save_json, resolve_device
import logging


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_prob = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_prob.append(probs)
        all_true.append(labels.numpy())
    return np.concatenate(all_true).astype(int), np.concatenate(all_prob)


def build_model_from_checkpoint(checkpoint: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = create_model(
        cfg["model"]["name"],
        in_channels=int(cfg["data"]["channels"]),
        num_classes=2,
        pretrained=False,
        dropout=float(cfg["model"].get("dropout", 0.2)),
    )
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return model.to(device)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    device = resolve_device(cfg.get("device", "auto"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    ckpt_cfg = checkpoint.get("config")
    if ckpt_cfg:
        cfg = ckpt_cfg
    model = build_model_from_checkpoint(checkpoint, cfg, device)

    data_root = Path(cfg["data"]["root"])
    x_path = data_root / cfg["data"][f"{args.split}_x"]
    y_path = data_root / cfg["data"][f"{args.split}_y"]
    ds = NpyPatchDataset(x_path, y_path, scale=cfg["data"]["scale"], augment=False)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    y_true, y_prob = predict_loader(model, loader, device)
    thresholds = np.linspace(float(cfg["thresholds"]["start"]), float(cfg["thresholds"]["stop"]), int(cfg["thresholds"]["steps"]))
    sweep = threshold_sweep(y_true, y_prob, thresholds)
    selected = args.threshold if args.threshold is not None else float(best_by(sweep, key="f1")["threshold"])
    metrics = binary_metrics(y_true, y_prob, selected)
    metrics["auc"] = auc_score(y_true, y_prob)
    metrics["ap"] = average_precision(y_true, y_prob)

    model_name = cfg["model"]["name"]
    out_dir = ensure_dir(Path(cfg["output_dir"]) / model_name)
    save_json(out_dir / f"{args.split}_metrics.json", metrics)
    save_csv(out_dir / f"{args.split}_threshold_sweep.csv", sweep, list(sweep[0].keys()))
    pred_rows = [
        {"index": i, "label": int(t), "prob_sw": float(p), "pred": int(p >= selected)}
        for i, (t, p) in enumerate(zip(y_true, y_prob))
    ]
    save_csv(out_dir / f"{args.split}_predictions.csv", pred_rows, ["index", "label", "prob_sw", "pred"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

