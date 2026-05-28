from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["_project_root"] = str(Path.cwd())
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="solid_waste_baselines/configs/baseline.yaml")
    parser.add_argument("--model", default=None, help="Override model.name in the config")
    parser.add_argument("--output-dir", default=None, help="Override output_dir in the config")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "model", None):
        cfg.setdefault("model", {})["name"] = args.model
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = args.output_dir
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    return cfg

