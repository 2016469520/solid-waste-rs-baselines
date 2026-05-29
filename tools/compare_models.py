from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging


DEFAULT_MODELS = [
    "small_cnn",
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "convnext_tiny",
]


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"Missing epoch column in {path}")
    return df.sort_values("epoch").reset_index(drop=True)


def load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def collect_results(root: Path, models: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    test_rows = []

    for model in models:
        model_dir = root / model
        history_path = model_dir / "history.csv"
        metrics_path = model_dir / "test_metrics.json"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        history = load_history(history_path)
        history["model"] = model
        history_rows.append(history)

        metrics = load_metrics(metrics_path)
        test_rows.append(
            {
                "model": model,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1"),
                "auc": metrics.get("auc"),
                "ap": metrics.get("ap"),
                "threshold": metrics.get("threshold"),
            }
        )

    history_df = pd.concat(history_rows, ignore_index=True)
    test_df = pd.DataFrame(test_rows)
    return history_df, test_df


def save_line_plot(history_df: pd.DataFrame, out_path: Path, value_col: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=history_df, x="epoch", y=value_col, hue="model", marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grouped_bar(test_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    plot_df = test_df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric")
    plt.title("Test Metrics Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_heatmap(test_df: pd.DataFrame, out_path: Path) -> None:
    heat_df = test_df.set_index("model")[["accuracy", "precision", "recall", "f1", "auc", "ap"]]
    plt.figure(figsize=(10, 4.8))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "Score"})
    plt.title("Test Metrics Heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_summary_csv(test_df: pd.DataFrame, out_path: Path) -> None:
    cols = ["model", "accuracy", "precision", "recall", "specificity", "f1", "auc", "ap", "threshold"]
    test_df[cols].to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model training and test results")
    parser.add_argument("--root", default="outputs_long20", help="Directory containing model subfolders")
    parser.add_argument("--output-dir", default="outputs_long20/compare_plots", help="Where plots will be written")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model names to compare")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    history_df, test_df = collect_results(root, args.models)
    save_summary_csv(test_df, out_dir / "test_summary.csv")

    save_line_plot(history_df, out_dir / "train_loss_by_model.png", "train_loss", "Train Loss", "Training Loss by Model")
    save_line_plot(history_df, out_dir / "val_f1_by_model.png", "val_f1", "Validation F1", "Validation F1 by Model")
    save_line_plot(history_df, out_dir / "val_accuracy_by_model.png", "val_accuracy", "Validation Accuracy", "Validation Accuracy by Model")
    save_grouped_bar(test_df, out_dir / "test_metrics_grouped_bar.png")
    save_heatmap(test_df, out_dir / "test_metrics_heatmap.png")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(f"Saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    main()