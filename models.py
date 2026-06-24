from pydantic import BaseModel
from typing import List, Dict


class Aspect(BaseModel):
    aspect_name: str
    description: str


class NuanceAttribute(BaseModel):
    dimension_name: str
    description: str
    possible_values: List[str]


class SyntheticReview(BaseModel):
    review: str
    aspects: List[str]
    aspect_scores: Dict[str, int]
    nuance_attributes: Dict[str, str]