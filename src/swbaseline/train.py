from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import add_common_args, apply_cli_overrides, load_config
from .datasets import NpyPatchDataset, dataset_summary, make_balanced_sampler
from .evaluate import predict_loader
from .metrics import average_precision, auc_score, best_by, binary_metrics, threshold_sweep
from .models import create_model
from .utils import checkpoint_payload, ensure_dir, save_csv, save_json, seed_everything, worker_init_fn, resolve_device


def build_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    data_root = Path(cfg["data"]["root"])
    return {
        "train_x": data_root / cfg["data"]["train_x"],
        "train_y": data_root / cfg["data"]["train_y"],
        "val_x": data_root / cfg["data"]["val_x"],
        "val_y": data_root / cfg["data"]["val_y"],
        "test_x": data_root / cfg["data"]["test_x"],
        "test_y": data_root / cfg["data"]["test_y"],
    }


def make_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    paths = build_paths(cfg)
    aug_cfg = cfg.get("augment", {})
    train_ds = NpyPatchDataset(
        paths["train_x"],
        paths["train_y"],
        scale=cfg["data"]["scale"],
        augment=bool(aug_cfg.get("enabled", True)),
        hflip_p=float(aug_cfg.get("hflip_p", 0.5)),
        vflip_p=float(aug_cfg.get("vflip_p", 0.5)),
        rotate90=bool(aug_cfg.get("rotate90", True)),
        noise_std=float(aug_cfg.get("noise_std", 0.0)),
    )
    val_ds = NpyPatchDataset(paths["val_x"], paths["val_y"], scale=cfg["data"]["scale"], augment=False)
    sampler = None
    shuffle = True
    if cfg["train"].get("class_balanced_sampler", True):
        sampler = make_balanced_sampler(np.asarray(train_ds.y))
        shuffle = False
    common = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["train"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def make_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    name = train_cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    train_cfg = cfg["train"]
    if train_cfg.get("scheduler", "cosine").lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
    return None


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp: bool) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_prob = []
    all_true = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(labels)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        all_prob.append(probs)
        all_true.append(labels.detach().cpu().numpy())
    y_prob = np.concatenate(all_prob)
    y_true = np.concatenate(all_true)
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = total_loss / max(len(y_true), 1)
    return metrics


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    seed_everything(int(cfg["seed"]))

    model_name = cfg["model"]["name"]
    run_dir = ensure_dir(Path(cfg["output_dir"]) / model_name)
    save_json(run_dir / "config.resolved.json", cfg)
    paths = build_paths(cfg)
    save_json(
        run_dir / "data_summary.json",
        {
            "train": dataset_summary(paths["train_x"], paths["train_y"]),
            "val": dataset_summary(paths["val_x"], paths["val_y"]),
            "test": dataset_summary(paths["test_x"], paths["test_y"]),
        },
    )

    device = resolve_device(cfg.get("device", "auto"))
    model = create_model(
        model_name,
        in_channels=int(cfg["data"]["channels"]),
        num_classes=2,
        pretrained=bool(cfg["model"].get("pretrained", True)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)
    train_loader, val_loader = make_loaders(cfg)
    train_labels = np.asarray(train_loader.dataset.y).astype(int)
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor([1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1)], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    thresholds = np.linspace(float(cfg["thresholds"]["start"]), float(cfg["thresholds"]["stop"]), int(cfg["thresholds"]["steps"]))
    monitor = cfg["train"].get("monitor", "val_f1")
    patience = int(cfg["train"].get("early_stop_patience", 12))
    best_score = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
        val_true, val_prob = predict_loader(model, val_loader, device)
        sweep = threshold_sweep(val_true, val_prob, thresholds)
        best_threshold_row = best_by(sweep, key="f1")
        val_metrics = binary_metrics(val_true, val_prob, float(best_threshold_row["threshold"]))
        val_metrics["auc"] = auc_score(val_true, val_prob)
        val_metrics["ap"] = average_precision(val_true, val_prob)
        val_metrics["best_threshold"] = float(best_threshold_row["threshold"])
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 3),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        save_csv(run_dir / "history.csv", history, list(history[0].keys()))
        save_csv(run_dir / "threshold_sweep_val.csv", sweep, list(sweep[0].keys()))

        score = float(row.get(monitor, row.get("val_f1", 0.0)))
        print(
            f"[{model_name}] epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} thr={val_metrics['best_threshold']:.4f}"
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            payload = checkpoint_payload(model, cfg, val_metrics, epoch)
            torch.save(payload, run_dir / "best.pt")
        else:
            bad_epochs += 1
        torch.save(checkpoint_payload(model, cfg, val_metrics, epoch), run_dir / "last.pt")
        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement.")
            break


if __name__ == "__main__":
    main()

