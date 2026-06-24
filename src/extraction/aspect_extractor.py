import json
from typing import List

from models import Aspect
from src.generation.llm_client import call_llm
from src.prompts.extraction_prompts import build_aspect_extraction_prompt


def parse_aspects(
    raw_output: str,
    min_aspects: int = 5,
    max_aspects: int = 8,
) -> List[Aspect]:
    # Parse the LLM JSON response.
    data = json.loads(raw_output)

    # JSON mode returns an object wrapper around the extracted list.
    if not isinstance(data, dict) or "aspects" not in data:
        raise ValueError("Aspect extraction output must contain an 'aspects' key")

    aspect_items = data["aspects"]

    # We expect a list of extracted aspects.
    if not isinstance(aspect_items, list):
        raise ValueError("'aspects' must be a list")

    # The model can choose how many aspects fit the document, within a fixed range.
    if not min_aspects <= len(aspect_items) <= max_aspects:
        raise ValueError(
            f"Expected between {min_aspects} and {max_aspects} aspects, "
            f"got {len(aspect_items)}"
        )

    # Validate each aspect against the schema.
    aspects = [Aspect(**item) for item in aspect_items]

    # Make sure each aspect maps to a unique score.
    aspect_names = [aspect.aspect_name for aspect in aspects]

    if len(set(aspect_names)) != len(aspect_names):
        raise ValueError("Duplicate aspect names found")

    return aspects


def extract_aspects_from_chunks(
    chunks: List[str],
    min_aspects: int = 5,
    max_aspects: int = 8,
    max_chunks: int = 5,
) -> List[Aspect]:
    # Limit the context to keep the prompt focused.
    product_text = "\n\n".join(chunks[:max_chunks])

    prompt = build_aspect_extraction_prompt(
        product_text=product_text,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
    )

    # Extract aspects from the product document using a stronger model,
    # because this step defines the schema for the whole dataset.
    raw_output = call_llm(
        prompt,
        model="gpt-4.1-mini",
        temperature=0,
    )

    return parse_aspects(
        raw_output=raw_output,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
    )