"""
Feature extraction module.
Builds TF-IDF matrix from sentences for downstream clustering and scoring.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_tfidf_matrix(sentences: list[str]) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Build a TF-IDF matrix from a list of sentences.
    Returns the matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer


def get_sentence_scores(tfidf_matrix) -> np.ndarray:
    """
    Score each sentence by the mean TF-IDF value of its terms.
    Higher score = more informative sentence.
    """
    scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
    return scores