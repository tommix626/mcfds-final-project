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


def heatmap(path: Path, matrix: np.ndarray, xlabels: list, ylabels: list, title: str, xlabel: str, ylabel: str, cbar_label: str) -> None:
    plt.figure(figsize=(6.4, 4.8))
    im = plt.imshow(matrix, aspect="auto", origin="lower")
    plt.xticks(np.arange(len(xlabels)), xlabels)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def eigenvalue_plot(path: Path, eigs_dict: dict[str, list[float]], title: str) -> None:
    plt.figure(figsize=(6.2, 4.4))
    for name, vals in eigs_dict.items():
        xs = np.arange(1, len(vals) + 1)
        plt.plot(xs, vals, marker="o", label=name)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Smallest eigenvalues of $L_{sym}$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def stacked_runtime(path: Path, df, title: str, baseline_df=None) -> None:
    """Stacked spectral runtime bars, optionally with a plain k-means baseline line."""
    plt.figure(figsize=(6.4, 4.4))
    x = np.arange(len(df))
    labels = [str(v) for v in df["n"].tolist()]
    bottom = np.zeros(len(df))
    for col, lab in [
        ("graph_seconds", "graph construction"),
        ("eigensolver_seconds", "eigensolver"),
        ("post_kmeans_seconds", "k-means in embedding"),
    ]:
        vals = df[col].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, label=lab)
        bottom += vals
    if baseline_df is not None and len(baseline_df) > 0:
        base = baseline_df.set_index("n").loc[df["n"].tolist()]
        plt.plot(x, base["total_seconds"].to_numpy(dtype=float), marker="D", linewidth=2, label="plain k-means baseline")
    plt.xticks(x, labels)
    plt.xlabel("Number of samples n")
    plt.ylabel("Seconds")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def multi_panel_scatter(path: Path, panels: list[tuple[np.ndarray, np.ndarray, str]], title: str) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.1))
    if n == 1:
        axes = [axes]
    for ax, (X, labels, subtitle) in zip(axes, panels):
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=12)
        ax.set_title(subtitle)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
