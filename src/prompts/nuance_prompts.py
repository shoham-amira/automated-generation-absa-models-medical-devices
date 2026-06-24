def build_nuance_extraction_prompt(product_text: str, n_nuances: int = 6) -> str:
    return f"""
You are extracting nuance attributes for generating written product reviews.

The following text is an official product document.

Your task is to define {n_nuances} nuance dimensions that can control how reviews are written and in what context.

A nuance attribute is not a product aspect.
It describes the reviewer background, usage context, writing style, or review situation.

Examples of useful dimensions include:
- who is writing the review
- experience level
- usage context
- workflow or usage conditions
- level of detail in writing
- type of review

Instructions:
1. Extract exactly {n_nuances} nuance dimensions.
2. Each dimension must be grounded in the document or reasonably implied by it.
3. Each dimension must affect how a written review would look.
4. Each dimension must help create realistic variation between reviews.
5. Avoid overlapping dimensions.
6. Use snake_case for dimension_name.
7. Keep names short and clear.
8. Focus only on written review characteristics.
9. Do not invent product claims, outcomes, risks, or benefits not supported by the document.

For each dimension, return:
- dimension_name
- description: short explanation of what this dimension represents
- possible_values: 3 to 6 realistic snake_case values

Output rules:
- Return ONLY valid JSON
- Return a JSON object with exactly one key: "nuances"
- "nuances" must be a list of exactly {n_nuances} objects
- Do not add explanations outside the JSON

Required JSON schema:
{{
  "nuances": [
    {{
      "dimension_name": "...",
      "description": "...",
      "possible_values": ["...", "...", "..."]
    }}
  ]
}}

Document text:
\"\"\"
{product_text}
\"\"\"
"""