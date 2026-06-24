import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (  # noqa: E402
    MODEL_COMPARISON_CONFIG,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_COMPARISON_PLOT_PATH,
)


METRIC_NAME_MAP = {
    "eval_element_accuracy": "Element Accuracy",
    "eval_exact_vector_match": "Exact Match",
    "eval_macro_f1": "Macro F1",
    "eval_precision": "Precision",
    "eval_recall": "Recall",
}


REQUIRED_METRICS = [
    "Element Accuracy",
    "Exact Match",
    "Macro F1",
    "Precision",
    "Recall",
]


MODEL_COLORS = {
    "Uroject12 - RoBERTa": "#6A8D92",
    "Uroject12 - DeBERTa": "#8FAF9F",
    "Medtronic 53401 - RoBERTa": "#C9A66B",
    "Medtronic 53401 - DeBERTa": "#B97A7A",
}


def load_metrics(metrics_path):
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        raw_metrics = json.load(file)

    normalized_metrics = {}

    for raw_key, pretty_key in METRIC_NAME_MAP.items():
        if raw_key not in raw_metrics:
            continue

        value = raw_metrics[raw_key]

        if isinstance(value, (int, float)) and value <= 1:
            value *= 100

        normalized_metrics[pretty_key] = round(value, 2)

    missing_metrics = [
        metric for metric in REQUIRED_METRICS
        if metric not in normalized_metrics
    ]

    if missing_metrics:
        raise KeyError(
            f"Missing metrics in {metrics_path}: {missing_metrics}\n"
            f"Available keys: {list(raw_metrics.keys())}"
        )

    return normalized_metrics


def build_comparison_dataframe():
    rows = []

    for item in MODEL_COMPARISON_CONFIG:
        metrics = load_metrics(item["metrics_path"])

        row = {
            "Device": item["device"],
            "Model": item["model"],
            "Device + Model": f'{item["device"]} - {item["model"]}',
            **metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_model_comparison(comparison_df):
    plot_df = comparison_df.set_index("Device + Model")[REQUIRED_METRICS].T

    colors = [
        MODEL_COLORS.get(model_name, "#888888")
        for model_name in plot_df.columns
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6.5),
        width=0.78,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
    )

    ax.set_title("Model Performance Comparison", fontsize=15, pad=16)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 105)

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Device and Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f",
            fontsize=8,
            padding=2,
        )

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    comparison_df = build_comparison_dataframe()

    comparison_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)
    plot_model_comparison(comparison_df)

    print(f"Saved comparison table to: {MODEL_COMPARISON_CSV_PATH}")
    print(f"Saved comparison plot to: {MODEL_COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    main()