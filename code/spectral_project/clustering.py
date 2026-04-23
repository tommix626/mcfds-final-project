from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph


@dataclass
class SpectralResult:
    labels: np.ndarray
    embedding: np.ndarray
    affinity: np.ndarray


def rbf_affinity(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    D2 = pairwise_distances(X, metric='sqeuclidean')
    if gamma is None:
        nonzero = D2[D2 > 0]
        sigma2 = np.median(nonzero) if nonzero.size else 1.0
        gamma = 1.0 / max(sigma2, 1e-8)
    W = np.exp(-gamma * D2)
    np.fill_diagonal(W, 0.0)
    return W


def knn_rbf_affinity(X: np.ndarray, n_neighbors: int = 10, gamma: float | None = None) -> np.ndarray:
    G = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)
    D = G.toarray()
    nz = D[D > 0]
    if gamma is None:
        sigma2 = np.median(nz**2) if nz.size else 1.0
        gamma = 1.0 / max(sigma2, 1e-8)
    W = np.zeros_like(D)
    mask = D > 0
    W[mask] = np.exp(-gamma * D[mask]**2)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    inv_sqrt = np.zeros_like(d)
    mask = d > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    Dm = np.diag(inv_sqrt)
    L = np.eye(W.shape[0]) - Dm @ W @ Dm
    return L


def graph_normalized_laplacian_sparse(W: csr_matrix) -> csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(d)
    mask = d > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    Dm = diags(inv_sqrt)
    I = diags(np.ones(W.shape[0]))
    return I - Dm @ W @ Dm


def ng_jordan_weiss(X: np.ndarray, n_clusters: int, n_neighbors: int = 10, random_state: int = 0) -> SpectralResult:
    W = knn_rbf_affinity(X, n_neighbors=n_neighbors)
    if W.shape[0] > 250:
        Ws = csr_matrix(W)
        Ls = graph_normalized_laplacian_sparse(Ws)
        vals, U = eigsh(Ls, k=n_clusters, which='SM')
    else:
        L = normalized_laplacian(W)
        vals, vecs = eigh(L)
        U = vecs[:, :n_clusters]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = U / norms
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(Y)
    return SpectralResult(labels=labels, embedding=Y, affinity=W)


def spectral_on_graph(W: np.ndarray, n_clusters: int, random_state: int = 0) -> SpectralResult:
    W = np.asarray(W)
    if W.shape[0] > 250:
        Ws = csr_matrix(W)
        Ls = graph_normalized_laplacian_sparse(Ws)
        vals, U = eigsh(Ls, k=n_clusters, which='SM')
    else:
        L = normalized_laplacian(W)
        vals, vecs = eigh(L)
        U = vecs[:, :n_clusters]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = U / norms
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(Y)
    return SpectralResult(labels=labels, embedding=Y, affinity=W)
