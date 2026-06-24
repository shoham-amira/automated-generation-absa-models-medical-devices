import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def load_aspect_names(aspects_path: Path) -> list[str]:
    """
    Load the aspect names from the aspect schema file.

    The order returned here defines the vector order used by all models.
    For example:
        index 0 -> ease_of_use
        index 1 -> durability_and_build_quality
    """

    with open(aspects_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract aspect names in the same order as they appear in aspects.json.
    return [item["aspect_name"] for item in data["aspects"]]


def scores_to_vector(
    aspect_scores: dict,
    aspect_names: list[str],
) -> list[int]:
    """
    Convert an aspect score dictionary into a fixed-order numeric vector.

    Input example:
        {
            "ease_of_use": 1,
            "durability_and_build_quality": 0
        }

    Output example:
        [1, 0, ...]

    The vector order is controlled by aspect_names.
    """

    # Build the output vector using the fixed aspect order.
    return [
        int(aspect_scores[aspect_name])
        for aspect_name in aspect_names
    ]


def prepare_aspect_score_dataframe(
    dataset_path: Path,
    aspect_names: list[str],
) -> pd.DataFrame:
    """
    Load the synthetic review dataset and prepare it for evaluation.

    Adds:
        model_input -> the review text used as model input
        labels      -> the ordered aspect-score vector used as ground truth

    This keeps the task format consistent:
        review text -> aspect-score vector
    """

    # Load JSONL dataset, where each line is one review record.
    df = pd.read_json(
        dataset_path,
        lines=True,
    )

    # The model input is the raw review text.
    df["model_input"] = df["review"]

    # Convert the aspect_scores dictionary into a vector label.
    df["labels"] = df["aspect_scores"].apply(
        lambda scores: scores_to_vector(
            aspect_scores=scores,
            aspect_names=aspect_names,
        )
    )

    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.30,
    validation_fraction_from_temp: float = 0.50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a reproducible train/validation/test split.

    Default behavior:
        70% train
        15% validation
        15% test

    This matches the split used by the RoBERTa model.
    """

    # First split: keep 70% for training and 30% for validation/test.
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    # Second split: divide the temporary set equally into validation and test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=validation_fraction_from_temp,
        random_state=random_state,
    )

    return train_df, val_df, test_df


def calculate_overall_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> dict:
    """
    Calculate global evaluation metrics across all aspects.

    The metric calculation flattens all aspect predictions, so each
    aspect score is treated as one prediction unit.

    Also calculates exact_vector_match, which checks whether the full
    aspect-score vector was predicted correctly for each review.
    """

    # Regression-style error metrics over the numeric score vectors.
    mae = mean_absolute_error(true_labels, pred_labels)
    mse = mean_squared_error(true_labels, pred_labels)
    rmse = np.sqrt(mse)

    # Flatten vectors to evaluate all aspect-level predictions together.
    flat_true = true_labels.flatten()
    flat_pred = pred_labels.flatten()

    # Element-level accuracy: correct aspect scores out of all aspect scores.
    element_accuracy = accuracy_score(flat_true, flat_pred)

    # Macro metrics over the three labels: -1, 0, 1.
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true,
        flat_pred,
        labels=[-1, 0, 1],
        average="macro",
        zero_division=0,
    )

    # Full-vector accuracy: all aspects in a review must be correct.
    exact_vector_match = np.mean(
        np.all(pred_labels == true_labels, axis=1)
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "element_accuracy": float(element_accuracy),
        "exact_vector_match": float(exact_vector_match),
        "macro_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def calculate_per_aspect_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    aspect_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate evaluation metrics separately for each aspect.

    This helps identify which aspects are easier or harder for the model.
    """

    per_aspect_rows = []
    per_aspect_confusions = {}

    # Evaluate each aspect independently by selecting its vector column.
    for i, aspect_name in enumerate(aspect_names):
        aspect_true = true_labels[:, i]
        aspect_pred = pred_labels[:, i]

        # Accuracy for this specific aspect.
        aspect_accuracy = accuracy_score(
            aspect_true,
            aspect_pred,
        )

        # Macro precision/recall/F1 for this aspect.
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_true,
            aspect_pred,
            labels=[-1, 0, 1],
            average="macro",
            zero_division=0,
        )

        # Confusion matrix for this aspect using fixed label order.
        cm = confusion_matrix(
            aspect_true,
            aspect_pred,
            labels=[-1, 0, 1],
        )

        # Store tabular per-aspect metrics.
        per_aspect_rows.append({
            "aspect": aspect_name,
            "accuracy": float(aspect_accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        })

        # Store confusion matrix separately as JSON-serializable data.
        per_aspect_confusions[aspect_name] = {
            "labels": ["negative", "not_mentioned", "positive"],
            "matrix": cm.tolist(),
        }

    return pd.DataFrame(per_aspect_rows), per_aspect_confusions