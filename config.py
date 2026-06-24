from pathlib import Path

# Change only these two values when running a different PDF
DATASET_NAME = "uroject12"
PDF_FILE_NAME = "uroject12-syringe-lever-instructions.pdf"

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_DIR = PROJECT_ROOT / "data" / DATASET_NAME
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
VALIDATION_DIR = BASE_DIR / "validation"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

ROBERTA_MODEL_DIR = MODELS_DIR / "roberta_aspect_scores"
ROBERTA_RESULTS_DIR = RESULTS_DIR / "roberta_aspect_scores"

# Input
PDF_PATH = RAW_DIR / PDF_FILE_NAME

# Processed outputs
EXTRACTED_TEXT_PATH = PROCESSED_DIR / "extracted_text.txt"
ASPECTS_PATH = PROCESSED_DIR / "aspects.json"
NUANCE_ATTRIBUTES_PATH = PROCESSED_DIR / "nuance_attributes.json"

# Main generated dataset
SYNTHETIC_REVIEWS_PATH = PROCESSED_DIR / "synthetic_reviews_2000.jsonl"
SYNTHETIC_REVIEWS_DEDUP_PATH = PROCESSED_DIR / "synthetic_reviews_2000_dedup.jsonl"

# Validation outputs
LLM_CONSISTENCY_AUDIT_PATH = VALIDATION_DIR / "llm_consistency_audit_results.jsonl"
MANUAL_REVIEW_SAMPLE_PATH = VALIDATION_DIR / "manual_review_sample.csv"
QUALITY_REPORT_PATH = VALIDATION_DIR / "quality_report.json"

# DeBERTa training outputs
DEBERTA_MODEL_DIR = MODELS_DIR / "deberta_aspect_scores"
DEBERTA_RESULTS_DIR = RESULTS_DIR / "deberta_aspect_scores"


# Create folders if missing
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ROBERTA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
ROBERTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEBERTA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEBERTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# This section is independent of DATASET_NAME.
# It is used only for comparing all trained models across all datasets.

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_RESULTS_DIR = DOCS_DIR / "results"

MODEL_COMPARISON_PLOT_PATH = DOCS_RESULTS_DIR / "model_performance_comparison.png"
MODEL_COMPARISON_CSV_PATH = DOCS_RESULTS_DIR / "model_performance_comparison.csv"

OVERALL_TEST_METRICS_FILE = "overall_test_metrics.json"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Used by src/visualization/plot_model_comparison.py.
# Add, remove, or edit entries here to control which device/model results
# appear in the comparison plot.
MODEL_COMPARISON_CONFIG = [
    {
        "device": "Uroject12",
        "model": "RoBERTa",
        "metrics_path": PROJECT_ROOT
        / "data"
        / "uroject12"
        / "results"
        / "roberta_aspect_scores"
        / OVERALL_TEST_METRICS_FILE,
    },
    {
        "device": "Uroject12",
        "model": "DeBERTa",
        "metrics_path": PROJECT_ROOT
        / "data"
        / "uroject12"
        / "results"
        / "deberta_aspect_scores"
        / OVERALL_TEST_METRICS_FILE,
    },
    {
        "device": "Medtronic 53401",
        "model": "RoBERTa",
        "metrics_path": PROJECT_ROOT
        / "data"
        / "MEDTRONIC_53401"
        / "results"
        / "roberta_aspect_scores"
        / OVERALL_TEST_METRICS_FILE,
    },
    {
        "device": "Medtronic 53401",
        "model": "DeBERTa",
        "metrics_path": PROJECT_ROOT
        / "data"
        / "MEDTRONIC_53401"
        / "results"
        / "deberta_aspect_scores"
        / OVERALL_TEST_METRICS_FILE,
    },
]