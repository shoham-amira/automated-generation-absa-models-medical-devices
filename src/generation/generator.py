import json
from typing import List
from pathlib import Path
from models import Aspect, NuanceAttribute, SyntheticReview
from src.generation.review_planner import sample_review_plan
from src.prompts.generation_prompts import build_review_generation_prompt
from src.generation.llm_client import call_llm
from src.prompts.consistency_check_prompt import build_consistency_check_prompt


def parse_generated_review(raw_json: str, review_plan: dict) -> SyntheticReview:
    # Parse the JSON returned by the LLM.
    data = json.loads(raw_json)

    if "review" not in data:
        raise ValueError("Generated output must contain a 'review' field")

    # Labels come from the controlled plan, not from the LLM.
    return SyntheticReview(
        review=data["review"],
        aspects=review_plan["aspects"],
        aspect_scores=review_plan["aspect_scores"],
        nuance_attributes=review_plan["nuance_attributes"],
    )


def generate_synthetic_reviews(
    product_context: str,
    aspects: List[Aspect],
    nuance_attributes: List[NuanceAttribute],
    n_reviews: int,
    output_file: str | Path,
):
    saved_count = 0
    attempted_count = 0
    max_total_attempts = n_reviews * 10

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:

        while saved_count < n_reviews and attempted_count < max_total_attempts:
            attempted_count += 1

            try:
                # Plan which aspects and nuance attributes this review should follow.
                review_plan = sample_review_plan(
                    aspects=aspects,
                    nuance_attributes=nuance_attributes,
                )

                # Selected aspects should always have a positive or negative score.
                for aspect_name in review_plan["aspects"]:
                    if review_plan["aspect_scores"][aspect_name] == 0:
                        raise ValueError(
                            f"Selected aspect has score 0: {aspect_name}"
                        )

                generation_prompt = build_review_generation_prompt(
                    product_context=product_context,
                    review_plan=review_plan,
                )

                review_saved = False

                for attempt in range(3):
                    # Generate the review text from the planned labels.
                    raw_output = call_llm(
                        generation_prompt,
                        model="gpt-4.1-nano",
                        temperature=0.5,
                    )

                    review = parse_generated_review(
                        raw_json=raw_output,
                        review_plan=review_plan,
                    )

                    consistency_prompt = build_consistency_check_prompt(
                        review_text=review.review,
                        review_plan=review_plan,
                    )

                    # Validate that the generated text matches the planned labels.
                    check_raw = call_llm(
                        consistency_prompt,
                        model="gpt-4.1-mini",
                        temperature=0,
                    )

                    check_result = json.loads(check_raw)
                    mismatches = check_result.get("mismatches", [])

                    if (
                        check_result.get("is_consistent") is True
                        and len(mismatches) == 0
                    ):
                        f.write(review.model_dump_json() + "\n")
                        saved_count += 1
                        review_saved = True
                        break

                    print(
                        f"Generated review attempt failed "
                        f"(saved {saved_count}/{n_reviews}, attempt {attempt + 1}/3):"
                    )
                    print(mismatches)

                if not review_saved:
                    print(
                        "Skipped one planned review after 3 inconsistent attempts. "
                        f"Continuing until {n_reviews} valid reviews are saved."
                    )

                if saved_count > 0 and saved_count % 100 == 0:
                    print(f"Saved {saved_count}/{n_reviews} valid reviews")

            except Exception as e:
                print(f"Generation attempt {attempted_count} failed:", e)

    if saved_count < n_reviews:
        print(
            f"Stopped after {attempted_count} attempts. "
            f"Saved {saved_count}/{n_reviews} valid reviews."
        )