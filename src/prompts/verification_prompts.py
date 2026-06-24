VERIFICATION_PROMPT = """
You are validating an aspect-based sentiment analysis (ABSA) label.

Check whether the sentiment label matches the sentiment expressed in the review specifically toward the given aspect.

Important:
- Focus only on the given aspect, not the overall review sentiment.
- The sentiment label can only be positive or negative.
- Positive means the review clearly praises the given aspect.
- Negative means the review clearly criticizes the given aspect.
- If the review does not clearly mention the given aspect, mark it inconsistent.
- If the review expresses both positive and negative sentiment toward the same aspect, mark it inconsistent.
- Ignore other aspects in the review unless they affect the sentiment toward the given aspect.

Return JSON only with this schema:
{{
  "is_consistent": true,
  "reason": "short explanation"
}}

Review:
\"\"\"
{review}
\"\"\"

Aspect:
{aspect}

Sentiment label:
{sentiment}
"""