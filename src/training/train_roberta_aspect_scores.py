import json
import sys
from pathlib import Path

import numpy as np

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import classification_report, confusion_matrix


# Allow imports from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    SYNTHETIC_REVIEWS_DEDUP_PATH,
    ASPECTS_PATH,
    ROBERTA_MODEL_DIR,
    ROBERTA_RESULTS_DIR,
)
from src.evaluation.aspect_score_utils import (
    load_aspect_names,
    prepare_aspect_score_dataframe,
    create_train_val_test_split,
    calculate_overall_metrics,
    calculate_per_aspect_metrics,
)

set_seed(42)


# Paths
DATASET_PATH = SYNTHETIC_REVIEWS_DEDUP_PATH
MODEL_OUTPUT_DIR = ROBERTA_MODEL_DIR
RESULTS_DIR = ROBERTA_RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_METRICS_FILE = RESULTS_DIR / "overall_test_metrics.json"
PER_ASPECT_METRICS_FILE = RESULTS_DIR / "per_aspect_metrics.csv"
PER_ASPECT_CONFUSION_FILE = RESULTS_DIR / "per_aspect_confusion_matrices.json"




def scores_to_labels(predictions: np.ndarray) -> np.ndarray:
    # Convert continuous regression outputs back to -1 / 0 / 1.
    labels = np.zeros_like(predictions, dtype=int)

    labels[predictions >= 0.5] = 1
    labels[predictions <= -0.5] = -1

    return labels


# LOAD DATASET

aspect_names = load_aspect_names(ASPECTS_PATH)
num_aspects = len(aspect_names)

print("Aspect order:")
for i, aspect in enumerate(aspect_names):
    print(f"{i}: {aspect}")

df = prepare_aspect_score_dataframe(
    dataset_path=DATASET_PATH,
    aspect_names=aspect_names,
)
# RoBERTa is trained as a regression model, so labels must be floats.
df["labels"] = df["labels"].apply(
    lambda vector: [float(value) for value in vector]
)
print(f"\nLoaded reviews: {len(df)}")
print(f"Number of aspects: {num_aspects}")


# TRAIN / VALIDATION / TEST SPLIT
train_df, val_df, test_df = create_train_val_test_split(df)

print("\nSplit sizes:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")


# CREATE HUGGINGFACE DATASETS

train_dataset = Dataset.from_pandas(
    train_df[["model_input", "labels"]],
    preserve_index=False,
)

val_dataset = Dataset.from_pandas(
    val_df[["model_input", "labels"]],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[["model_input", "labels"]],
    preserve_index=False,
)


# MODEL AND TOKENIZER

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(example):
    # Tokenize review text.
    return tokenizer(
        example["model_input"],
        truncation=True,
        max_length=256,
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Add moderate dropout to reduce memorization on synthetic data.
model_config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_aspects,
    problem_type="regression",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
    classifier_dropout=0.2,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=model_config,
)


# METRICS
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    pred_labels = scores_to_labels(predictions)
    true_labels = labels.astype(int)

    return calculate_overall_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
    )

# TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir=str(RESULTS_DIR),

    # Conservative fine-tuning for a small synthetic dataset.
    learning_rate=1e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # Fewer epochs reduce memorization risk.
    num_train_epochs=3,

    # Stronger regularization than the previous run.
    weight_decay=0.05,

    logging_steps=25,

    eval_strategy="epoch",
    save_strategy="epoch",

    # Use validation loss because F1 was already near-perfect.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_total_limit=1,

    report_to="none",

    fp16=False,
)


# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=1,
        )
    ],
)


# TRAIN

trainer.train()


# FINAL EVALUATION

results = trainer.evaluate(test_dataset)

print("\nFINAL TEST RESULTS")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")


# TEST PREDICTIONS

predictions = trainer.predict(test_dataset)

pred_continuous = predictions.predictions
pred_labels = scores_to_labels(pred_continuous)

true_labels = predictions.label_ids.astype(int)

flat_pred = pred_labels.flatten()
flat_true = true_labels.flatten()


# OVERALL CLASSIFICATION REPORT

print("\nFLATTENED CLASSIFICATION REPORT")
print(
    classification_report(
        flat_true,
        flat_pred,
        labels=[-1, 0, 1],
        target_names=["negative", "not_mentioned", "positive"],
        zero_division=0,
    )
)

print("\nFLATTENED CONFUSION MATRIX")
print(
    confusion_matrix(
        flat_true,
        flat_pred,
        labels=[-1, 0, 1],
    )
)


# PER-ASPECT METRICS
per_aspect_df, per_aspect_confusions = calculate_per_aspect_metrics(
    true_labels=true_labels,
    pred_labels=pred_labels,
    aspect_names=aspect_names,
)

print("\nPER-ASPECT DETAILED METRICS")
print(per_aspect_df)

# SAVE EVALUATION REPORTS

overall_metrics = {
    key: float(value)
    for key, value in results.items()
    if isinstance(value, float)
}

with open(OVERALL_METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        overall_metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )

per_aspect_df.to_csv(
    PER_ASPECT_METRICS_FILE,
    index=False,
    encoding="utf-8-sig",
)

with open(PER_ASPECT_CONFUSION_FILE, "w", encoding="utf-8") as f:
    json.dump(
        per_aspect_confusions,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nSaved evaluation reports:")
print(f"Overall metrics: {OVERALL_METRICS_FILE}")
print(f"Per-aspect metrics: {PER_ASPECT_METRICS_FILE}")
print(f"Per-aspect confusion matrices: {PER_ASPECT_CONFUSION_FILE}")


# SAVE MODEL

trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

# Save aspect order for inference.
with open(MODEL_OUTPUT_DIR / "aspect_order.json", "w", encoding="utf-8") as f:
    json.dump(
        {"aspect_order": aspect_names},
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nModel saved successfully!")
print(f"Saved to: {MODEL_OUTPUT_DIR}")