import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Allow imports from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Select the model results directory.
# To switch models, change both the import and the variable below:
# DeBERTa: from config import DEBERTA_RESULTS_DIR
# RoBERTa: from config import ROBERTA_RESULTS_DIR
from config import DEBERTA_RESULTS_DIR
RESULTS_DIR = DEBERTA_RESULTS_DIR


PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PER_ASPECT_METRICS_FILE = RESULTS_DIR / "per_aspect_metrics.csv"


def plot_per_aspect_macro_f1():
    # Load per-aspect evaluation metrics.
    df = pd.read_csv(PER_ASPECT_METRICS_FILE)

    # Sort aspects by Macro-F1 for easier interpretation.
    df = df.sort_values("macro_f1", ascending=False)

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["aspect"],
        df["macro_f1"],
    )

    plt.title("Per-Aspect Macro-F1")
    plt.xlabel("Aspect")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = PLOTS_DIR / "per_aspect_macro_f1.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    plot_per_aspect_macro_f1()
    print("Per-aspect Macro-F1 plot created successfully.")


if __name__ == "__main__":
    main()