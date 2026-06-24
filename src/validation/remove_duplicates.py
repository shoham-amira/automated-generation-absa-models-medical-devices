import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from config import (
    SYNTHETIC_REVIEWS_PATH,
    SYNTHETIC_REVIEWS_DEDUP_PATH,
)

def main():
    # Keep track of review texts we already saved
    seen_reviews = set()

    total = 0
    kept = 0
    removed = 0

    # Make sure the output folder exists
    SYNTHETIC_REVIEWS_DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SYNTHETIC_REVIEWS_PATH, "r", encoding="utf-8") as input_f, \
         open(SYNTHETIC_REVIEWS_DEDUP_PATH, "w", encoding="utf-8") as output_f:

        for line in input_f:
            total += 1

            # Parse one JSONL row
            item = json.loads(line)

            # Use the review text as the duplicate key
            review_text = item["review"].strip()

            # Skip exact duplicate reviews
            if review_text in seen_reviews:
                removed += 1
                continue

            # Save first occurrence only
            seen_reviews.add(review_text)
            output_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print("=== DUPLICATE REMOVAL REPORT ===")
    print(f"Input reviews: {total}")
    print(f"Kept reviews: {kept}")
    print(f"Removed duplicates: {removed}")
    print(f"Output file: {SYNTHETIC_REVIEWS_DEDUP_PATH}")


if __name__ == "__main__":
    main()