import fitz
import re
import unicodedata
from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    full_text = ""

    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text += page.get_text() + "\n"

    return full_text


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("•", "-")
    text = text.replace("/uni00A0", " ")
    text = unicodedata.normalize("NFKC", text)

    return text.strip()


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_with_period = sentence + ". "

        if len(current_chunk) + len(sentence_with_period) <= max_chars:
            current_chunk += sentence_with_period
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = sentence_with_period

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks