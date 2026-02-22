"""
Text preprocessing module.
Handles tokenization, stopword removal, and text cleaning.
"""

import re
import ssl
import nltk

# Fix SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """Remove extra whitespace and normalize the text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences."""
    ensure_nltk_data()
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_and_clean(sentence: str, remove_stopwords: bool = True) -> list[str]:
    """
    Tokenize a sentence into words, lowercase, remove non-alpha tokens,
    and optionally remove stopwords.
    """
    ensure_nltk_data()
    tokens = word_tokenize(sentence.lower())
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]
    return tokens


def preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    Full preprocessing pipeline.
    Returns cleaned text and list of sentences.
    """
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    return cleaned, sentences