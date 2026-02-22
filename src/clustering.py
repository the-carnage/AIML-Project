"""
Sentence clustering module.
Groups sentences by topic using K-Means, then picks the best from each cluster.
"""

import numpy as np
from sklearn.cluster import KMeans


def cluster_sentences(tfidf_matrix, n_clusters: int) -> np.ndarray:
    """
    Cluster sentences using K-Means on TF-IDF features.
    Returns cluster labels for each sentence.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)
    return labels


def select_representative_sentences(
    sentences: list[str],
    labels: np.ndarray,
    sentence_scores: np.ndarray,
) -> list[str]:
    """
    From each cluster, select the sentence with the highest TF-IDF score.
    Returns the selected sentences in their original document order.
    """
    selected_indices = []
    for cluster_id in range(labels.max() + 1):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        best_idx = cluster_indices[np.argmax(sentence_scores[cluster_indices])]
        selected_indices.append(best_idx)

    # Maintain original order
    selected_indices.sort()
    return [sentences[i] for i in selected_indices]