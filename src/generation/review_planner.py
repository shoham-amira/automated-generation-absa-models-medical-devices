import random
from typing import List

from models import Aspect, NuanceAttribute


def sample_aspect_score() -> int:
    # Selected aspects are written as either positive or negative.
    return random.choice([-1, 1])


def sample_review_plan(
    aspects: List[Aspect],
    nuance_attributes: List[NuanceAttribute],
    min_aspects: int = 1,
    max_aspects: int = 2,
    min_nuances: int = 2,
    max_nuances: int = 5,
) -> dict:

    n_aspects = random.randint(min_aspects, min(max_aspects, len(aspects)))
    n_nuances = random.randint(min_nuances, min(max_nuances, len(nuance_attributes)))

    selected_aspects = random.sample(aspects, n_aspects)
    selected_nuances = random.sample(nuance_attributes, n_nuances)

    # Build the full score vector: 0 means the aspect is not mentioned.
    aspect_scores = {
        aspect.aspect_name: 0
        for aspect in aspects
    }

    # Only selected aspects receive positive or negative sentiment.
    for aspect in selected_aspects:
        aspect_scores[aspect.aspect_name] = sample_aspect_score()

    sampled_nuance_values = {
        nuance.dimension_name: random.choice(nuance.possible_values)
        for nuance in selected_nuances
    }

    return {
        "aspects": [aspect.aspect_name for aspect in selected_aspects],
        "aspect_scores": aspect_scores,
        "nuance_attributes": sampled_nuance_values,
    }