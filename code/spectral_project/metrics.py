import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    C = np.zeros((classes.size, clusters.size), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            C[i, j] = np.sum((y_true == c) & (y_pred == k))
    r, c = linear_sum_assignment(C.max() - C)
    return C[r, c].sum() / len(y_true)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for k in np.unique(y_pred):
        idx = y_pred == k
        vals, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / len(y_true)


def summarize(y_true, y_pred):
    return {
        'accuracy': float(clustering_accuracy(y_true, y_pred)),
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
        'purity': float(purity_score(y_true, y_pred)),
    }
