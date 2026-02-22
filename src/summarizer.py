"""
Main summarizer module.
Orchestrates the full extractive summarization pipeline:
  preprocess → feature extraction → clustering → summary
"""

from src.preprocess import preprocess_text
from src.feature_extraction import build_tfidf_matrix, get_sentence_scores
from src.clustering import cluster_sentences, select_representative_sentences


def summarize(text: str, ratio: float = 0.3) -> dict:
    """
    Summarize input text using extractive summarization.

    Args:
        text:  The input text to summarize.
        ratio: Fraction of sentences to keep (0.0–1.0).

    Returns:
        A dict with keys:
            - summary: the generated summary string
            - original_sentence_count: number of sentences in input
            - summary_sentence_count: number of sentences in summary
            - compression_ratio: how much the text was compressed
    """
    cleaned, sentences = preprocess_text(text)

    if len(sentences) <= 2:
        return {
            "summary": cleaned,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(sentences),
            "compression_ratio": 1.0,
        }

    # Determine number of clusters (= summary sentences)
    n_clusters = max(1, int(len(sentences) * ratio))
    n_clusters = min(n_clusters, len(sentences))

    # Build features
    tfidf_matrix, _ = build_tfidf_matrix(sentences)
    scores = get_sentence_scores(tfidf_matrix)

    # Cluster and pick representatives
    labels = cluster_sentences(tfidf_matrix, n_clusters)
    summary_sentences = select_representative_sentences(sentences, labels, scores)

    summary_text = " ".join(summary_sentences)

    return {
        "summary": summary_text,
        "original_sentence_count": len(sentences),
        "summary_sentence_count": len(summary_sentences),
        "compression_ratio": round(len(summary_sentences) / len(sentences), 2),
    }