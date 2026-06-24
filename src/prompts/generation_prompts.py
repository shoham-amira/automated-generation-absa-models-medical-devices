def build_review_generation_prompt(product_context: str, review_plan: dict) -> str:
    selected_aspects = review_plan["aspects"]

    selected_aspect_scores = {
        aspect: review_plan["aspect_scores"][aspect]
        for aspect in selected_aspects
    }

    return f"""
You are generating one short synthetic product review for an ABSA dataset.

The review must match the planned aspect labels exactly.

SELECTED ASPECTS AND REQUIRED SENTIMENT:
{selected_aspect_scores}

Score meaning:
- 1 = clearly positive opinion about this aspect
- -1 = clearly negative opinion about this aspect

RULES:
- Mention only the selected aspects.
- Do not mention, imply, praise, criticize, or describe any unselected aspect.
- Each selected aspect must be clearly mentioned.
- Each selected aspect must match its required sentiment.
- Do not express both positive and negative sentiment toward the same aspect.
- Do not add an overall summary or general impression.
- If one aspect is selected, write one focused sentence.
- If multiple aspects are selected, write one clear phrase or sentence for each aspect.
- Avoid broad wording that could imply another aspect unless that aspect is selected.
- Do not use positive wording for an aspect with score -1.
- Do not use negative wording for an aspect with score 1.
- Do not mention aspects with score 0.

STYLE:
- Sound like a real intended user writing a short online review.
- Write naturally and briefly.
- Do not copy aspect names mechanically.
- Use nuance attributes only as background for tone and perspective.
- Do not explicitly mention the nuance attributes.
- Keep the review 1 to 3 sentences.

Before returning, check silently:
- Are all selected aspects included?
- Do all sentiments match?
- Are unselected aspects completely absent?

PRODUCT CONTEXT:
\"\"\"
{product_context}
\"\"\"

NUANCE ATTRIBUTES:
{review_plan["nuance_attributes"]}

Return ONLY valid JSON:
{{
  "review": "..."
}}
"""