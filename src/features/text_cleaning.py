"""Text cleaning shared by training AND serving, so both see identical input.

Keeping this in one small module (no pandas/sklearn imports) prevents
training-serving skew and keeps the API image light.
"""
import re

HTML_TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove HTML tags (IMDB reviews contain <br />) and collapse whitespace.

    Case and punctuation are kept on purpose: transformer tokenizers use them,
    and the TF-IDF baseline lowercases on its own.
    """
    text = HTML_TAG.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()