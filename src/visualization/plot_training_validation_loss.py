import json
import sys
from pathlib import Path

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


def find_trainer_state_file() -> Path:
    # Find the latest HuggingFace checkpoint folder.
    checkpoint_dirs = sorted(
        RESULTS_DIR.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )

    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No checkpoint folders found in: {RESULTS_DIR}"
        )

    trainer_state_path = checkpoint_dirs[-1] / "trainer_state.json"

    if not trainer_state_path.exists():
        raise FileNotFoundError(
            f"trainer_state.json not found in: {trainer_state_path}"
        )

    return trainer_state_path


def load_log_history(trainer_state_path: Path) -> list[dict]:
    # Load Trainer logs saved during training.
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    return trainer_state["log_history"]

def extract_avg_training_loss_by_epoch(log_history: list[dict]) -> tuple[list[int], list[float]]:
    # Average training loss values within each epoch.
    epoch_losses = {}

    for log in log_history:
        if "loss" in log and "epoch" in log:
            epoch_number = int(log["epoch"]) + 1

            if epoch_number not in epoch_losses:
                epoch_losses[epoch_number] = []

            epoch_losses[epoch_number].append(log["loss"])

    epochs = sorted(epoch_losses.keys())
    avg_losses = [
        sum(epoch_losses[epoch]) / len(epoch_losses[epoch])
        for epoch in epochs
    ]

    return epochs, avg_losses


def extract_validation_loss_by_epoch(log_history: list[dict]) -> tuple[list[int], list[float]]:
    # Extract validation loss after each epoch.
    eval_logs = [
        log for log in log_history
        if "eval_loss" in log and "epoch" in log
    ]

    epochs = [int(log["epoch"]) for log in eval_logs]
    losses = [log["eval_loss"] for log in eval_logs]

    return epochs, losses


def plot_training_validation_loss(log_history: list[dict]):
    # Plot one training-loss point and one validation-loss point per epoch.
    train_epochs, train_losses = extract_avg_training_loss_by_epoch(log_history)
    eval_epochs, eval_losses = extract_validation_loss_by_epoch(log_history)

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_epochs,
        train_losses,
        marker="o",
        label="Average Training Loss",
    )

    plt.plot(
        eval_epochs,
        eval_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.title("Training and Validation Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(eval_epochs)
    plt.legend()
    plt.tight_layout()

    output_path = PLOTS_DIR / "training_validation_loss_by_epoch.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    trainer_state_path = find_trainer_state_file()
    log_history = load_log_history(trainer_state_path)

    plot_training_validation_loss(log_history)

    print("Training history plot created successfully.")
    print(f"Trainer state file: {trainer_state_path}")


if __name__ == "__main__":
    main()