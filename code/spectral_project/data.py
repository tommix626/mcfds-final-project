from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn import datasets
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def _scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)



def make_success_datasets(random_state: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray, int, str]]:
    rs = np.random.default_rng(random_state)
    X1, y1 = datasets.make_moons(n_samples=500, noise=0.08, random_state=random_state)
    X2, y2 = datasets.make_circles(n_samples=500, factor=0.45, noise=0.06, random_state=random_state)
    X3, y3 = datasets.make_blobs(
        n_samples=600,
        centers=[(-4, -1), (0, 3), (4, -1)],
        cluster_std=[0.9, 1.0, 0.8],
        random_state=random_state,
    )
    X4_base, y4 = datasets.make_moons(n_samples=600, noise=0.07, random_state=random_state)
    noise = rs.normal(scale=0.35, size=(X4_base.shape[0], 18))
    X4 = np.hstack([X4_base, noise])
    X5, y5 = datasets.make_blobs(
        n_samples=800,
        centers=4,
        n_features=50,
        cluster_std=2.0,
        center_box=(-12.0, 12.0),
        random_state=random_state,
    )
    return {
        "two_moons": (X1, y1, 2, "Low-dimensional, non-convex manifolds; spectral should clearly beat k-means."),
        "circles": (X2, y2, 2, "Concentric rings violate the convex-cluster assumption of k-means."),
        "blobs": (X3, y3, 3, "Well-separated convex clusters; both methods should do well."),
        "high_dim_moons": (X4, y4, 2, "Two moons embedded in 20D with noise coordinates; spectral should still exploit neighborhood geometry."),
        "high_dim_blobs": (X5, y5, 4, "High-dimensional but well-separated blobs; both methods should remain strong."),
    }



def make_failure_datasets(random_state: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray, int, str]]:
    rs = np.random.default_rng(random_state)
    X1, y1 = datasets.make_blobs(
        n_samples=600,
        centers=[[0.0, 0.0], [0.9, 0.9], [1.8, 1.8]],
        cluster_std=[1.0, 1.0, 1.0],
        random_state=random_state,
    )

    left = rs.normal(loc=[-2.5, 0.0], scale=[0.45, 0.45], size=(220, 2))
    right = rs.normal(loc=[2.5, 0.0], scale=[0.45, 0.45], size=(220, 2))
    bridge_x = np.linspace(-2.0, 2.0, 160)
    bridge_y = rs.normal(loc=0.0, scale=0.12, size=bridge_x.shape[0])
    bridge = np.c_[bridge_x, bridge_y]
    X2 = np.vstack([left, right, bridge])
    y2 = np.concatenate([
        np.zeros(left.shape[0], dtype=int),
        np.ones(right.shape[0], dtype=int),
        np.zeros(bridge.shape[0], dtype=int),
    ])

    return {
        "overlapping_blobs_failure": (X1, y1, 3, "Heavy overlap makes any graph partition ambiguous; both methods should degrade."),
        "bridge_failure": (X2, y2, 2, "A thin bridge creates a near-connection between clusters, so normalized cuts can merge them."),
    }



def load_iris_data() -> tuple[np.ndarray, np.ndarray]:
    ds = load_iris()
    return _scale(ds.data), ds.target



def load_digits_subset(classes=(0, 1, 3, 6), random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ds = load_digits()
    X = ds.data
    y = ds.target
    mask = np.isin(y, np.array(classes))
    X = X[mask]
    y = y[mask]
    # Shuffle for repeated experiments but preserve all examples of requested classes.
    rs = np.random.default_rng(random_state)
    perm = rs.permutation(len(y))
    return _scale(X[perm]), y[perm]



def digits_pca_2d(classes=(0, 1, 3, 6), random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_digits_subset(classes=classes, random_state=random_state)
    X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return X2, y



def load_all_digits() -> tuple[np.ndarray, np.ndarray]:
    ds = load_digits()
    return _scale(ds.data), ds.target



def karate_graph() -> tuple[nx.Graph, np.ndarray]:
    G = nx.karate_club_graph()
    y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes()])
    return G, y



def stochastic_block_model_graph(random_state: int = 0) -> tuple[nx.Graph, np.ndarray]:
    sizes = [45, 45, 45]
    p = [[0.22, 0.02, 0.02], [0.02, 0.22, 0.02], [0.02, 0.02, 0.22]]
    G = nx.stochastic_block_model(sizes, p, seed=random_state)
    labels = np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])
    return G, labels
