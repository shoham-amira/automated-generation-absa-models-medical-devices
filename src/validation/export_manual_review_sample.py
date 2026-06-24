import json
import pandas as pd
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from config import (
    SYNTHETIC_REVIEWS_DEDUP_PATH,
    MANUAL_REVIEW_SAMPLE_PATH,
)


INPUT_FILE = SYNTHETIC_REVIEWS_DEDUP_PATH
OUTPUT_FILE = MANUAL_REVIEW_SAMPLE_PATH


def score_to_sentiment(score: int) -> str:
    # Convert numeric label to readable text
    if score == 1:
        return "positive"
    if score == -1:
        return "negative"
    if score == 0:
        return "not_mentioned"
    return "invalid"


rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        aspect_scores = item["aspect_scores"]

        # Keep only selected aspects with their labels
        selected_labels = {
            aspect: score_to_sentiment(aspect_scores[aspect])
            for aspect in item["aspects"]
        }

        rows.append({
            "review": item["review"],
            "selected_aspects": ", ".join(item["aspects"]),
            "selected_labels": json.dumps(selected_labels, ensure_ascii=False),
            "aspect_scores": json.dumps(aspect_scores, ensure_ascii=False),
            "nuance_attributes": json.dumps(item["nuance_attributes"], ensure_ascii=False),
        })


df = pd.DataFrame(rows)

sample_df = df.sample(
    n=100,
    random_state=42,
)

# Create validation folder if it does not exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

sample_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(f"Sample exported successfully: {OUTPUT_FILE}")