from __future__ import annotations

import argparse
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader

from .config import add_common_args, apply_cli_overrides, load_config
from .datasets import PatchFolderDataset
from .evaluate import build_model_from_checkpoint
from .utils import load_checkpoint, save_csv, resolve_device


@torch.no_grad()
def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    device = resolve_device(cfg.get("device", "auto"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("config"):
        cfg = checkpoint["config"]
    model = build_model_from_checkpoint(checkpoint, cfg, device)
    ds = PatchFolderDataset(args.patch_dir, scale=cfg["data"]["scale"])
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    model.eval()
    for images, filenames in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        for filename, prob in zip(filenames, probs):
            rows.append(
                {
                    "filename": filename,
                    "prob_sw": float(prob),
                    "pred": int(float(prob) >= args.threshold),
                    "threshold": float(args.threshold),
                }
            )
    save_csv(Path(args.output), rows, ["filename", "prob_sw", "pred", "threshold"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

