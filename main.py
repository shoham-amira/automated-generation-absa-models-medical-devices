import json
from models import Aspect, NuanceAttribute
from src.preprocessing.pdf_utils import extract_text_from_pdf, clean_text, chunk_text
from src.extraction.aspect_extractor import extract_aspects_from_chunks
from src.extraction.nuance_extractor import extract_nuances_from_chunks
from src.generation.generator import generate_synthetic_reviews
from config import (
    PDF_PATH,
    EXTRACTED_TEXT_PATH,
    ASPECTS_PATH,
    NUANCE_ATTRIBUTES_PATH,
    SYNTHETIC_REVIEWS_PATH,
)

N_REVIEWS = 2000

MIN_ASPECTS = 5
MAX_ASPECTS = 8

N_NUANCES = 6


def save_aspects(aspects, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {"aspects": [aspect.model_dump() for aspect in aspects]},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_aspects(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [Aspect(**item) for item in data["aspects"]]


def save_nuances(nuance_attributes, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {"nuance_attributes": [nuance.model_dump() for nuance in nuance_attributes]},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_nuances(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [NuanceAttribute(**item) for item in data["nuance_attributes"]]


def main():
    # Read and prepare the product document.
    pdf_text = extract_text_from_pdf(PDF_PATH)
    cleaned_text = clean_text(pdf_text)
    chunks = chunk_text(cleaned_text)

    print(f"Number of chunks: {len(chunks)}")
    print("Preview:")
    print(cleaned_text[:500])

    # Load or create aspect schema.
    if ASPECTS_PATH.exists():
        aspects = load_aspects(ASPECTS_PATH)
        print("Loaded aspects from file")
    else:
        aspects = extract_aspects_from_chunks(
            chunks=chunks,
            min_aspects=MIN_ASPECTS,
            max_aspects=MAX_ASPECTS,
        )
        save_aspects(aspects, ASPECTS_PATH)
        print("Extracted and saved aspects")

    # Load or create nuance schema.
    if NUANCE_ATTRIBUTES_PATH.exists():
        nuance_attributes = load_nuances(NUANCE_ATTRIBUTES_PATH)
        print("Loaded nuance attributes from file")
    else:
        nuance_attributes = extract_nuances_from_chunks(
            chunks=chunks,
            n_nuances=N_NUANCES,
        )
        save_nuances(nuance_attributes, NUANCE_ATTRIBUTES_PATH)
        print("Extracted and saved nuance attributes")

    print(f"Using {len(aspects)} aspects")
    print(f"Using {len(nuance_attributes)} nuance attributes")

    print("Aspects:")
    for aspect in aspects:
        print(f"- {aspect.aspect_name}: {aspect.description}")

    generate_synthetic_reviews(
        product_context=cleaned_text[:6000],
        aspects=aspects,
        nuance_attributes=nuance_attributes,
        n_reviews=N_REVIEWS,
        output_file=SYNTHETIC_REVIEWS_PATH,
    )


if __name__ == "__main__":
    main()