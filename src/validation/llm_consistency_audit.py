import json
import os
import random
from collections import defaultdict, Counter
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from config import (
    SYNTHETIC_REVIEWS_DEDUP_PATH,
    LLM_CONSISTENCY_AUDIT_PATH,
)
from dotenv import load_dotenv
from openai import OpenAI
from src.prompts.verification_prompts import VERIFICATION_PROMPT


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = SYNTHETIC_REVIEWS_DEDUP_PATH
OUTPUT_FILE = LLM_CONSISTENCY_AUDIT_PATH


# Number of examples sampled for each aspect-sentiment pair
SAMPLES_PER_GROUP = 20

MODEL_NAME = "gpt-4.1-mini"
RANDOM_SEED = 42


def score_to_sentiment(score: int) -> str:
    # Convert numeric label to text label for the verifier
    if score == 1:
        return "positive"
    if score == -1:
        return "negative"

    raise ValueError(f"Invalid selected aspect score: {score}")


def load_review_level_rows() -> list[dict]:
    # Load the generated JSONL reviews
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for review_id, line in enumerate(f, start=1):
            item = json.loads(line)
            item["review_id"] = review_id
            rows.append(item)

    return rows


def flatten_selected_aspects(rows: list[dict]) -> list[dict]:
    # Convert review-level rows into ABSA-style audit rows
    flattened_rows = []

    for row in rows:
        review_text = row["review"]
        aspect_scores = row["aspect_scores"]

        for aspect in row["aspects"]:
            score = aspect_scores[aspect]

            # Only selected aspects should be audited
            if score == 0:
                continue

            flattened_rows.append({
                "review_id": row["review_id"],
                "text": review_text,
                "aspect": aspect,
                "sentiment": score_to_sentiment(score),
                "score": score,
            })

    return flattened_rows


def stratified_sample(rows: list[dict]) -> list[dict]:
    # Sample evenly from each aspect-sentiment group
    grouped = defaultdict(list)

    for row in rows:
        key = (row["aspect"], row["sentiment"])
        grouped[key].append(row)

    sampled = []

    random.seed(RANDOM_SEED)

    for key, group_rows in grouped.items():
        sample_size = min(SAMPLES_PER_GROUP, len(group_rows))
        sampled.extend(random.sample(group_rows, sample_size))

    return sampled


def verify_row(row: dict) -> dict:
    # Ask the LLM whether the label matches the review text
    prompt = VERIFICATION_PROMPT.format(
        review=row["text"],
        aspect=row["aspect"],
        sentiment=row["sentiment"],
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def main():
    review_rows = load_review_level_rows()
    absa_rows = flatten_selected_aspects(review_rows)
    sampled_rows = stratified_sample(absa_rows)

    print(f"Loaded reviews: {len(review_rows)}")
    print(f"Flattened ABSA rows for audit: {len(absa_rows)}")
    print(f"Sampled rows: {len(sampled_rows)}")

    results_counter = Counter()
    group_counter = Counter()

    # Make sure validation folder exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_f:
        for i, row in enumerate(sampled_rows, start=1):
            try:
                result = verify_row(row)

                is_consistent = bool(result.get("is_consistent"))
                reason = result.get("reason", "")

                output_item = {
                    "review_id": row["review_id"],
                    "aspect": row["aspect"],
                    "sentiment": row["sentiment"],
                    "score": row["score"],
                    "is_consistent": is_consistent,
                    "reason": reason,
                    "text": row["text"],
                }

                output_f.write(
                    json.dumps(output_item, ensure_ascii=False) + "\n"
                )

                results_counter[
                    "consistent" if is_consistent else "inconsistent"
                ] += 1

                group_counter[
                    (row["aspect"], row["sentiment"], is_consistent)
                ] += 1

                print(f"Checked {i}/{len(sampled_rows)}")

            except Exception as e:
                results_counter["errors"] += 1
                print(f"Error on row {i}: {e}")

    print("\n=== LLM CONSISTENCY AUDIT REPORT ===")
    print(results_counter)

    total_checked = (
        results_counter["consistent"]
        + results_counter["inconsistent"]
    )

    if total_checked > 0:
        consistency_rate = results_counter["consistent"] / total_checked
        print(f"Consistency rate: {consistency_rate:.2%}")

    print("\n=== GROUP SUMMARY ===")
    for key, value in group_counter.items():
        print(key, value)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()