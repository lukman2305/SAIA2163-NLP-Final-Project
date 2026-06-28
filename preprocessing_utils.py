"""
Multilingual preprocessing utilities for the SAIA2163 NLP Final Project.

This version is designed for the full multilingual dataset:
- EN: English
- FR: French
- DE: German
- JP: Japanese
- IT: Italian

Important design choice:
Do NOT use English-only lemmatization or English stopword removal for multilingual data.
Those steps can damage French, German, Italian, and Japanese text. Instead, this file uses
light Unicode-safe cleaning and optional emoji conversion.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

try:
    import emoji  # type: ignore
except Exception:  # pragma: no cover
    emoji = None

SUPPORTED_LANGUAGES: dict[str, str] = {
    "EN": "English",
    "FR": "French",
    "DE": "German",
    "JP": "Japanese",
    "IT": "Italian",
}

LABEL_MAPPING: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INVERSE_LABEL_MAPPING: dict[int, str] = {v: k for k, v in LABEL_MAPPING.items()}
LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]


def normalize_language_code(value: Any) -> str:
    """Normalize language values such as ' EN' into 'EN'."""
    if value is None:
        return "UNKNOWN"
    code = str(value).strip().upper()
    if not code or code in {"NAN", "NONE", "NULL"}:
        return "UNKNOWN"
    return code


def language_name(code: Any) -> str:
    """Return a readable language name for a normalized language code."""
    code = normalize_language_code(code)
    return SUPPORTED_LANGUAGES.get(code, code.title())


def add_language_token(text: Any, language: Any) -> str:
    """
    Add a simple language token so traditional ML models can learn language-specific patterns.

    Example:
        add_language_token("This app is good", "EN")
        -> "lang_en This app is good"
    """
    code = normalize_language_code(language).lower()
    if code == "unknown":
        code = "unknown"
    return f"lang_{code} {'' if text is None else str(text)}"


def map_score_to_label(score: Any) -> str:
    """Map Google Play star score into 3 sentiment labels."""
    try:
        score_value = float(score)
    except Exception:
        return "neutral"

    if score_value <= 2:
        return "negative"
    if score_value == 3:
        return "neutral"
    return "positive"


def _demojize(text: str) -> str:
    """Convert emoji into words when the optional emoji package is installed."""
    if emoji is None:
        return text
    try:
        return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    except Exception:
        return text


def multilingual_light_clean_text(text: Any) -> str:
    """
    Unicode-safe cleaning for multilingual TF-IDF / character n-gram models.

    This intentionally avoids English-only stemming, lemmatization, and stopword removal.
    It keeps non-Latin scripts such as Japanese instead of deleting them.
    """
    if text is None:
        return ""

    text = str(text)
    if not text.strip():
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _demojize(text)
    text = text.lower()

    # Remove URLs, emails, and HTML tags but keep Unicode letters/numbers/scripts.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep words, numbers, underscores, and most Unicode text. Replace only noisy punctuation.
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[|•●◆★☆]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transformer_clean_text(text: Any) -> str:
    """
    Light cleaning for multilingual transformer input.

    Transformer models such as XLM-R or multilingual BERT should receive natural text,
    not stemmed or lemmatized text.
    """
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = _demojize(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Backward-compatible aliases so older code that imports these names still runs.
preprocess_text = multilingual_light_clean_text
light_clean_text = transformer_clean_text
