import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import SYNTHETIC_REVIEWS_DEDUP_PATH


INPUT_FILE = SYNTHETIC_REVIEWS_DEDUP_PATH


def count_sentences(text: str) -> int:
    # Simple sentence counter based on punctuation.
    endings = [".", "!", "?"]
    count = sum(text.count(end) for end in endings)

    # Every non-empty review counts as at least one sentence.
    return max(count, 1)


def main():
    word_count_counter = Counter()
    sentence_count_counter = Counter()
    selected_aspect_count_counter = Counter()
    duplicate_counter = Counter()

    total = 0
    word_counts = []
    char_counts = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            total += 1

            review = item["review"].strip()
            aspects = item["aspects"]

            # Basic text length metrics.
            words = review.split()
            word_count = len(words)
            char_count = len(review)
            sentence_count = count_sentences(review)

            word_counts.append(word_count)
            char_counts.append(char_count)

            # Group reviews into length buckets.
            if word_count < 10:
                word_count_counter["under_10_words"] += 1
            elif word_count <= 25:
                word_count_counter["10_to_25_words"] += 1
            elif word_count <= 50:
                word_count_counter["26_to_50_words"] += 1
            else:
                word_count_counter["over_50_words"] += 1

            sentence_count_counter[sentence_count] += 1
            selected_aspect_count_counter[len(aspects)] += 1

            # Used later for duplicate detection.
            duplicate_counter[review.lower()] += 1

    duplicate_review_texts = sum(
        1 for count in duplicate_counter.values()
        if count > 1
    )

    duplicate_extra_rows = sum(
        count - 1 for count in duplicate_counter.values()
        if count > 1
    )

    print("=== REVIEW DISTRIBUTION REPORT ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Total reviews: {total}")

    if total == 0:
        print("No reviews found.")
        return

    print("\n=== WORD COUNT ===")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Average words: {sum(word_counts) / len(word_counts):.2f}")
    print(f"Word count buckets: {word_count_counter}")

    print("\n=== CHARACTER COUNT ===")
    print(f"Min chars: {min(char_counts)}")
    print(f"Max chars: {max(char_counts)}")
    print(f"Average chars: {sum(char_counts) / len(char_counts):.2f}")

    print("\n=== SENTENCE COUNT ===")
    print(sentence_count_counter)

    print("\n=== SELECTED ASPECT COUNT ===")
    print(selected_aspect_count_counter)

    print("\n=== DUPLICATES ===")
    print(f"Duplicate review texts: {duplicate_review_texts}")
    print(f"Duplicate extra rows: {duplicate_extra_rows}")

    print("\n=== TOP DUPLICATES ===")
    for review, count in duplicate_counter.most_common(10):
        if count > 1:
            print(f"{count}x - {review}")


if __name__ == "__main__":
    main()