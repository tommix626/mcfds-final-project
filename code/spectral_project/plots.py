from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def scatter_clusters(path: Path, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    plt.figure(figsize=(5.2, 4.2))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=14)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()



def line_noise(path: Path, xs, spectral_scores, kmeans_scores, title: str, ylabel: str) -> None:
    plt.figure(figsize=(5.2, 4.1))
    plt.plot(xs, spectral_scores, marker="o", label="Spectral clustering")
    plt.plot(xs, kmeans_scores, marker="s", label="k-means")
    plt.xlabel("Noise standard deviation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()



def line_scaling(path: Path, xs, ys_dict: dict[str, list[float]], title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5.4, 4.2))
    for name, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()



def bar_comparison(path: Path, categories: list[str], values_a: list[float], values_b: list[float], label_a: str, label_b: str, title: str, ylabel: str) -> None:
    x = np.arange(len(categories))
    width = 0.38
    plt.figure(figsize=(max(6.5, 0.7 * len(categories)), 4.2))
    plt.bar(x - width / 2, values_a, width=width, label=label_a)
    plt.bar(x + width / 2, values_b, width=width, label=label_b)
    plt.xticks(x, categories, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
