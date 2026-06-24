def build_consistency_check_prompt(review_text: str, review_plan: dict) -> str:
    selected_aspects = review_plan["aspects"]

    selected_aspect_scores = {
        aspect: review_plan["aspect_scores"][aspect]
        for aspect in selected_aspects
    }

    return f"""
You are validating whether one synthetic review matches its planned aspect-level sentiment labels.

Your job is to find ONLY mismatches between the review and the planned labels.
Do not report correct matches.
Do not explain aspects that match correctly.

Review:
\"\"\"
{review_text}
\"\"\"

Selected aspects and required sentiment:
{selected_aspect_scores}

All aspect scores:
{review_plan["aspect_scores"]}

Score meaning:
- 1 = the review expresses a clearly positive opinion about this aspect
- -1 = the review expresses a clearly negative opinion about this aspect
- 0 = this aspect should not be mentioned or clearly implied in the review

There are no neutral or mixed labels in this dataset.

A mismatch exists ONLY if:
- A selected aspect is missing from the review.
- A selected aspect has the opposite sentiment from its required score.
- A selected aspect has both positive and negative sentiment.
- An aspect with score 0 is mentioned, implied, praised, criticized, or described.

Important:
- If an aspect matches its required sentiment, do NOT include it.
- If an aspect is correctly absent because its score is 0, do NOT include it.
- The mismatches list must contain ONLY real mismatches.
- If there are no real mismatches, return is_consistent true and an empty mismatches list.

Return ONLY valid JSON.

If consistent, return exactly:
{{
  "is_consistent": true,
  "mismatches": []
}}

If inconsistent, return:
{{
  "is_consistent": false,
  "mismatches": [
    {{
      "aspect": "...",
      "problem": "short explanation of the mismatch"
    }}
  ]
}}
"""