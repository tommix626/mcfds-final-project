from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


@dataclass
class SpectralResult:
    labels: np.ndarray
    embedding: np.ndarray
    affinity: csr_matrix | np.ndarray
    eigenvalues: np.ndarray
    timings: dict[str, float] | None = None
    memory_bytes: dict[str, int] | None = None


@dataclass
class KMeansResult:
    labels: np.ndarray
    timings: dict[str, float]



def _sparse_memory_bytes(A: csr_matrix) -> int:
    A = A.tocsr()
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)



def knn_rbf_affinity_sparse(
    X: np.ndarray,
    n_neighbors: int = 10,
    gamma: float | None = None,
    include_self: bool = False,
) -> csr_matrix:
    """Build a symmetric sparse kNN RBF affinity matrix."""
    G = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
        include_self=include_self,
        metric="minkowski",
        p=2,
    ).tocsr()
    if G.nnz == 0:
        raise ValueError("kNN graph is empty; increase n_neighbors or check the dataset.")

    if gamma is None:
        sigma2 = float(np.median(np.square(G.data))) if G.data.size else 1.0
        gamma = 1.0 / max(sigma2, 1e-12)

    W = G.copy().astype(float)
    W.data = np.exp(-gamma * np.square(W.data))
    W = W.maximum(W.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W



def normalized_laplacian_sparse(W: csr_matrix) -> csr_matrix:
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(d)
    mask = d > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    Dm = diags(inv_sqrt)
    I = diags(np.ones(W.shape[0]))
    return (I - Dm @ W @ Dm).tocsr()



def normalized_laplacian_dense(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    inv_sqrt = np.zeros_like(d)
    mask = d > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    Dm = np.diag(inv_sqrt)
    return np.eye(W.shape[0]) - Dm @ W @ Dm



def _topk_smallest_eigenvectors(L: csr_matrix | np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if issparse(L):
        vals, U = eigsh(L, k=k, which="SM")
        order = np.argsort(vals)
        return vals[order], U[:, order]
    vals, U = eigh(L)
    return vals[:k], U[:, :k]



def row_normalize(U: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return U / norms



def ng_jordan_weiss(
    X: np.ndarray,
    n_clusters: int,
    n_neighbors: int = 10,
    gamma: float | None = None,
    random_state: int = 0,
    n_init: int = 20,
    dense_threshold: int = 220,
) -> SpectralResult:
    timings: dict[str, float] = {}
    memory: dict[str, int] = {"X": int(X.nbytes)}

    t0 = perf_counter()
    W_sparse = knn_rbf_affinity_sparse(X, n_neighbors=n_neighbors, gamma=gamma)
    timings["graph_seconds"] = perf_counter() - t0
    memory["affinity_sparse"] = _sparse_memory_bytes(W_sparse)
    memory["nnz_affinity"] = int(W_sparse.nnz)

    t1 = perf_counter()
    if X.shape[0] <= dense_threshold:
        W = W_sparse.toarray()
        memory["affinity_dense"] = int(W.nbytes)
        L = normalized_laplacian_dense(W)
        vals, U = _topk_smallest_eigenvectors(L, n_clusters)
        affinity = W
    else:
        L = normalized_laplacian_sparse(W_sparse)
        vals, U = _topk_smallest_eigenvectors(L, n_clusters)
        affinity = W_sparse
    timings["eigensolver_seconds"] = perf_counter() - t1

    Y = row_normalize(U)
    memory["embedding"] = int(Y.nbytes)

    t2 = perf_counter()
    labels = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(Y)
    timings["kmeans_seconds"] = perf_counter() - t2
    timings["total_seconds"] = timings["graph_seconds"] + timings["eigensolver_seconds"] + timings["kmeans_seconds"]

    return SpectralResult(
        labels=labels,
        embedding=Y,
        affinity=affinity,
        eigenvalues=vals,
        timings=timings,
        memory_bytes=memory,
    )



def kmeans_baseline(X: np.ndarray, n_clusters: int, random_state: int = 0, n_init: int = 20) -> KMeansResult:
    t0 = perf_counter()
    labels = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(X)
    return KMeansResult(labels=labels, timings={"kmeans_seconds": perf_counter() - t0})



def spectral_on_graph(W: np.ndarray | csr_matrix, n_clusters: int, random_state: int = 0, n_init: int = 20) -> SpectralResult:
    timings: dict[str, float] = {}
    if not issparse(W):
        W_arr = np.asarray(W, dtype=float)
        W_arr = np.maximum(W_arr, W_arr.T)
        np.fill_diagonal(W_arr, 0.0)
        memory = {"affinity_dense": int(W_arr.nbytes)}
    else:
        W_arr = W.maximum(W.T).tocsr()
        W_arr.setdiag(0.0)
        W_arr.eliminate_zeros()
        memory = {"affinity_sparse": _sparse_memory_bytes(W_arr), "nnz_affinity": int(W_arr.nnz)}

    t1 = perf_counter()
    if issparse(W_arr):
        L = normalized_laplacian_sparse(W_arr)
        vals, U = _topk_smallest_eigenvectors(L, n_clusters)
    else:
        L = normalized_laplacian_dense(W_arr)
        vals, U = _topk_smallest_eigenvectors(L, n_clusters)
    timings["eigensolver_seconds"] = perf_counter() - t1

    Y = row_normalize(U)
    memory["embedding"] = int(Y.nbytes)
    t2 = perf_counter()
    labels = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(Y)
    timings["kmeans_seconds"] = perf_counter() - t2
    timings["total_seconds"] = timings["eigensolver_seconds"] + timings["kmeans_seconds"]

    return SpectralResult(
        labels=labels,
        embedding=Y,
        affinity=W_arr,
        eigenvalues=vals,
        timings=timings,
        memory_bytes=memory,
    )
