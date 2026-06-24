def build_aspect_extraction_prompt(
    product_text: str,
    min_aspects: int = 5,
    max_aspects: int = 8,
) -> str:
    return f"""
You are an expert in aspect-based sentiment analysis (ABSA).

Extract between {min_aspects} and {max_aspects} reviewable aspects from the product document below.
These aspects will be used to generate and label synthetic product reviews.

Choose the number of aspects based on the document:
- Use fewer aspects if only a few clear review topics are supported.
- Use more aspects only if the document supports distinct, non-overlapping review topics.
- Do not force extra aspects just to reach the maximum.

A good aspect:
- Represents one clear review topic, not a combination of separate topics
- Is grounded in the document
- Describes something an intended user could realistically praise or complain about in a short review
- Can receive both positive and negative opinions
- Is practical and concrete
- Has a clear meaning that does not strongly overlap with another aspect

Avoid aspects that:
- Combine separate review topics into one aspect
- Use "and" to combine two review topics unless they are truly inseparable in natural review language
- Are too technical, rare, narrow, administrative, regulatory, or document-specific
- Refer to package identifiers, symbols, legal restrictions, or manufacturer-only details
- Are only evidence, symptoms, indicators, or sub-parts of a broader aspect
- Duplicate another aspect in meaning, even if the wording is different

Duplicate prevention:
- Each aspect must represent a distinct review topic.
- If two aspects would usually be described with the same sentence, merge them.
- If one aspect mainly helps judge another broader aspect, keep only the broader aspect.
- If an aspect name needs "and", check whether it is combining separate topics. If so, choose one clearer topic or merge it into a broader single concept.

Grounding rule:
Every aspect must be supported by the document.
Do not invent features, benefits, risks, or use cases not described in the document.

Output rules:
Return ONLY valid JSON.
No explanation, no markdown.

For each aspect provide:
- aspect_name: short snake_case label
- description: one sentence describing what users would comment on

Required schema:
{{
  "aspects": [
    {{
      "aspect_name": "...",
      "description": "..."
    }}
  ]
}}

Return between {min_aspects} and {max_aspects} aspects.

Product document:
\"\"\"
{product_text}
\"\"\"
"""