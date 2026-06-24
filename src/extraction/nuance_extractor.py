import json
from typing import List

from models import NuanceAttribute
from src.generation.llm_client import call_llm
from src.prompts.nuance_prompts import build_nuance_extraction_prompt


def parse_nuances(raw_output: str, expected_count: int) -> List[NuanceAttribute]:
    # Parse the LLM JSON response.
    data = json.loads(raw_output)

    # JSON mode returns an object wrapper around the extracted list.
    if not isinstance(data, dict) or "nuances" not in data:
        raise ValueError("Nuance extraction output must contain a 'nuances' key")

    nuance_items = data["nuances"]

    # We expect a list of extracted nuance dimensions.
    if not isinstance(nuance_items, list):
        raise ValueError("'nuances' must be a list")

    # Keep a fixed number of nuances for consistent review generation.
    if len(nuance_items) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} nuances, got {len(nuance_items)}")

    # Validate each nuance against the schema.
    nuances = [NuanceAttribute(**item) for item in nuance_items]

    # Avoid duplicated nuance dimensions.
    dimension_names = [nuance.dimension_name for nuance in nuances]

    if len(set(dimension_names)) != len(dimension_names):
        raise ValueError("Duplicate nuance dimension names found")

    return nuances


def extract_nuances_from_chunks(
    chunks: List[str],
    n_nuances: int = 6,
    max_chunks: int = 5,
) -> List[NuanceAttribute]:
    # Limit the context to keep the prompt focused.
    product_text = "\n\n".join(chunks[:max_chunks])

    prompt = build_nuance_extraction_prompt(
        product_text=product_text,
        n_nuances=n_nuances,
    )

    # Extract review context dimensions from the product document.
    raw_output = call_llm(
        prompt,
        model="gpt-4.1-mini",
        temperature=0,
    )

    return parse_nuances(
        raw_output=raw_output,
        expected_count=n_nuances,
    )