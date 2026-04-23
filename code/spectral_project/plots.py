import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def scatter_clusters(path: Path, X: np.ndarray, labels: np.ndarray, title: str, y_true=None):
    plt.figure(figsize=(5,4))
    plt.scatter(X[:,0], X[:,1], c=labels, s=15)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def line_noise(path: Path, sigmas, spectral_scores, kmeans_scores, title: str, ylabel: str):
    plt.figure(figsize=(5,4))
    plt.plot(sigmas, spectral_scores, marker='o', label='Spectral clustering')
    plt.plot(sigmas, kmeans_scores, marker='s', label='k-means')
    plt.xlabel('Noise standard deviation')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
