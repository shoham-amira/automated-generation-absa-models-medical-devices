import json
from collections import Counter
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    SYNTHETIC_REVIEWS_DEDUP_PATH,
    QUALITY_REPORT_PATH,
)


INPUT_FILE = SYNTHETIC_REVIEWS_DEDUP_PATH
OUTPUT_REPORT_FILE = QUALITY_REPORT_PATH


REQUIRED_FIELDS = [
    "review",
    "aspects",
    "aspect_scores",
    "nuance_attributes",
]

VALID_SCORES = {-1, 0, 1}

FORBIDDEN_PHRASES = [
    "you should take",
    "you must stop",
    "guaranteed cure",
    "completely safe for everyone",
    "no need to consult a doctor",
    "start taking",
    "stop taking",
    "change your medication",
]


def score_to_sentiment(score: int) -> str:
    if score == 1:
        return "positive"
    if score == -1:
        return "negative"
    if score == 0:
        return "not_mentioned"
    return "invalid"


def quality_check(item: dict) -> tuple[bool, list[str]]:
    errors = []

    # Check required top-level fields.
    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"missing_field:{field}")

    if errors:
        return False, errors

    review_text = item["review"]

    if not isinstance(review_text, str):
        errors.append("review_not_string")
        return False, errors

    text = review_text.strip()

    if len(text) < 30:
        errors.append("review_too_short")

    if len(text) > 1500:
        errors.append("review_too_long")

    aspects = item["aspects"]
    aspect_scores = item["aspect_scores"]
    nuance_attributes = item["nuance_attributes"]

    if not isinstance(aspects, list):
        errors.append("aspects_not_list")
        return False, errors

    if not isinstance(aspect_scores, dict):
        errors.append("aspect_scores_not_dict")
        return False, errors

    if not isinstance(nuance_attributes, dict):
        errors.append("nuance_attributes_not_dict")

    if not aspects:
        errors.append("no_selected_aspects")

    # Selected aspects must exist in the score vector and cannot have score 0.
    for aspect in aspects:
        if aspect not in aspect_scores:
            errors.append(f"selected_aspect_missing_from_scores:{aspect}")
        elif aspect_scores[aspect] == 0:
            errors.append(f"selected_aspect_has_zero_score:{aspect}")

    # All scores must be valid labels.
    for aspect, score in aspect_scores.items():
        if score not in VALID_SCORES:
            errors.append(f"invalid_score:{aspect}:{score}")

    selected_aspects_set = set(aspects)

    # Selected aspects should be non-zero; unselected aspects should be zero.
    for aspect, score in aspect_scores.items():
        if aspect in selected_aspects_set and score == 0:
            errors.append(f"selected_aspect_zero:{aspect}")

        if aspect not in selected_aspects_set and score != 0:
            errors.append(f"unselected_aspect_nonzero:{aspect}:{score}")

    lowered = text.lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            errors.append(f"forbidden_phrase:{phrase}")

    return len(errors) == 0, errors


def main():
    total = 0
    valid = 0
    invalid = 0

    error_counter = Counter()
    aspect_counter = Counter()
    score_counter = Counter()
    aspect_score_counter = Counter()
    selected_aspect_count_counter = Counter()
    duplicate_counter = Counter()

    all_score_keys = None
    inconsistent_score_keys = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                error_counter["json_decode_error"] += 1
                continue

            is_valid, errors = quality_check(item)

            if is_valid:
                valid += 1
            else:
                invalid += 1
                for error in errors:
                    error_counter[error] += 1

            review_text = item.get("review", "").strip()
            duplicate_counter[review_text] += 1

            aspects = item.get("aspects", [])
            aspect_scores = item.get("aspect_scores", {})

            if not isinstance(aspects, list) or not isinstance(aspect_scores, dict):
                continue

            selected_aspect_count_counter[len(aspects)] += 1

            # All rows should use the same product-specific aspect schema.
            current_keys = tuple(aspect_scores.keys())

            if all_score_keys is None:
                all_score_keys = current_keys
            elif current_keys != all_score_keys:
                inconsistent_score_keys += 1

            for aspect in aspects:
                aspect_counter[aspect] += 1

                score = aspect_scores.get(aspect)

                if score in VALID_SCORES:
                    sentiment = score_to_sentiment(score)
                    score_counter[sentiment] += 1
                    aspect_score_counter[(aspect, sentiment)] += 1
                else:
                    score_counter["invalid"] += 1

    duplicate_review_texts = sum(
        1 for count in duplicate_counter.values()
        if count > 1
    )

    duplicate_extra_rows = sum(
        count - 1 for count in duplicate_counter.values()
        if count > 1
    )

    report = {
        "input_file": str(INPUT_FILE),
        "total_reviews": total,
        "valid_reviews": valid,
        "invalid_reviews": invalid,
        "duplicate_review_texts": duplicate_review_texts,
        "duplicate_extra_rows": duplicate_extra_rows,
        "aspect_score_vector_keys": list(all_score_keys) if all_score_keys else [],
        "rows_with_inconsistent_aspect_score_keys": inconsistent_score_keys,
        "selected_aspect_count_distribution": dict(selected_aspect_count_counter),
        "aspect_distribution": dict(aspect_counter),
        "score_distribution": dict(score_counter),
        "errors": dict(error_counter),
        "aspect_score_distribution": {
            f"{aspect}|{sentiment}": count
            for (aspect, sentiment), count in aspect_score_counter.items()
        },
    }

    OUTPUT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== QUALITY CHECK REPORT ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Total reviews: {total}")
    print(f"Valid reviews: {valid}")
    print(f"Invalid reviews: {invalid}")

    print("\n=== DUPLICATES ===")
    print(f"Duplicate review texts: {duplicate_review_texts}")
    print(f"Duplicate extra rows: {duplicate_extra_rows}")

    print("\n=== SCHEMA CONSISTENCY ===")
    print(f"Aspect-score vector keys: {all_score_keys}")
    print(f"Rows with inconsistent aspect-score keys: {inconsistent_score_keys}")

    print("\n=== SELECTED ASPECT COUNT DISTRIBUTION ===")
    print(selected_aspect_count_counter)

    print("\n=== ASPECT DISTRIBUTION ===")
    print(aspect_counter)

    print("\n=== ASPECT-LEVEL SENTIMENT DISTRIBUTION ===")
    print(score_counter)

    print("\n=== ERRORS ===")
    print(error_counter)

    print(f"\nSaved quality report to: {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    main()