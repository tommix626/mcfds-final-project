from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


@dataclass
class SpectralResult:
    labels: np.ndarray
    embedding: np.ndarray
    affinity: csr_matrix | np.ndarray
    eigenvalues: np.ndarray
    timings: dict[str, float] | None = None
    memory_bytes: dict[str, int] | None = None
    parameters: dict[str, float | int | str] | None = None


@dataclass
class KMeansResult:
    labels: np.ndarray
    timings: dict[str, float]



def _sparse_memory_bytes(A: csr_matrix) -> int:
    A = A.tocsr()
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)


def gamma_from_sigma(sigma: float) -> float:
    """Convert the paper's RBF scale sigma into exp(-gamma * distance^2)."""
    return 1.0 / (2.0 * max(float(sigma) ** 2, 1e-12))


def median_pairwise_distance(X: np.ndarray) -> float:
    D = pairwise_distances(X)
    vals = D[D > 1e-12]
    return float(np.median(vals)) if vals.size else 1.0


def median_knn_distance(X: np.ndarray, n_neighbors: int) -> float:
    G = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
        include_self=False,
        metric="minkowski",
        p=2,
    ).tocsr()
    return float(np.median(G.data)) if G.data.size else 1.0



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


def knn_rbf_affinity_sigma_sparse(
    X: np.ndarray,
    n_neighbors: int = 10,
    sigma: float = 1.0,
) -> csr_matrix:
    """Build a symmetric sparse kNN RBF affinity using exp(-d^2 / (2 sigma^2))."""
    return knn_rbf_affinity_sparse(X, n_neighbors=n_neighbors, gamma=gamma_from_sigma(sigma))



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
        v0 = np.linspace(1.0, 2.0, L.shape[0])
        vals, U = eigsh(L, k=k, which="SM", v0=v0)
        order = np.argsort(vals)
        return vals[order], U[:, order]
    vals, U = eigh(L)
    return vals[:k], U[:, :k]



def row_normalize(U: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return U / norms


def embedding_distortion(Y: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lab in np.unique(labels):
        pts = Y[labels == lab]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0, keepdims=True)
        total += float(np.sum((pts - center) ** 2))
    return total


def _spectral_embedding_from_sparse_affinity(
    W_sparse: csr_matrix,
    n_clusters: int,
    dense_threshold: int,
) -> tuple[np.ndarray, np.ndarray, csr_matrix | np.ndarray, dict[str, int]]:
    memory: dict[str, int] = {
        "affinity_sparse": _sparse_memory_bytes(W_sparse),
        "nnz_affinity": int(W_sparse.nnz),
    }
    if W_sparse.shape[0] <= dense_threshold:
        W = W_sparse.toarray()
        memory["affinity_dense"] = int(W.nbytes)
        L = normalized_laplacian_dense(W)
        vals, U = _topk_smallest_eigenvectors(L, n_clusters)
        return vals, row_normalize(U), W, memory
    L = normalized_laplacian_sparse(W_sparse)
    vals, U = _topk_smallest_eigenvectors(L, n_clusters)
    return vals, row_normalize(U), W_sparse, memory


def select_sigma_by_embedding_distortion(
    X: np.ndarray,
    n_clusters: int,
    n_neighbors: int = 10,
    alphas: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1.40, 2.00, 3.00, 5.00),
    random_state: int = 0,
    n_init: int = 20,
    dense_threshold: int = 220,
) -> dict:
    base = median_knn_distance(X, n_neighbors=n_neighbors)
    best: dict | None = None
    for alpha in alphas:
        sigma = max(alpha * base, 1e-12)
        t_graph = perf_counter()
        W_sparse = knn_rbf_affinity_sigma_sparse(X, n_neighbors=n_neighbors, sigma=sigma)
        graph_seconds = perf_counter() - t_graph
        try:
            t_eig = perf_counter()
            vals, Y, affinity, memory = _spectral_embedding_from_sparse_affinity(W_sparse, n_clusters, dense_threshold)
            eig_seconds = perf_counter() - t_eig
            t_km = perf_counter()
            labels = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(Y)
            km_seconds = perf_counter() - t_km
            distortion = embedding_distortion(Y, labels)
        except Exception:
            continue
        candidate = {
            "alpha": float(alpha),
            "base_distance": float(base),
            "sigma": float(sigma),
            "gamma": gamma_from_sigma(sigma),
            "distortion": float(distortion),
            "labels": labels,
            "embedding": Y,
            "affinity": affinity,
            "eigenvalues": vals,
            "memory": memory,
            "graph_seconds": graph_seconds,
            "eigensolver_seconds": eig_seconds,
            "kmeans_seconds": km_seconds,
        }
        if best is None or candidate["distortion"] < best["distortion"]:
            best = candidate
    if best is None:
        raise RuntimeError("sigma search failed for all candidate scales")
    return best



def ng_jordan_weiss(
    X: np.ndarray,
    n_clusters: int,
    n_neighbors: int = 10,
    gamma: float | None = None,
    random_state: int = 0,
    n_init: int = 20,
    dense_threshold: int = 220,
) -> SpectralResult:
    memory: dict[str, int] = {"X": int(X.nbytes)}

    if gamma is None:
        t_search = perf_counter()
        best = select_sigma_by_embedding_distortion(
            X,
            n_clusters=n_clusters,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_init=n_init,
            dense_threshold=dense_threshold,
        )
        timings = {
            "graph_seconds": best["graph_seconds"],
            "eigensolver_seconds": best["eigensolver_seconds"],
            "kmeans_seconds": best["kmeans_seconds"],
            "sigma_search_seconds": perf_counter() - t_search,
        }
        timings["total_seconds"] = timings["sigma_search_seconds"]
        memory.update(best["memory"])
        Y = best["embedding"]
        labels = best["labels"]
        vals = best["eigenvalues"]
        affinity = best["affinity"]
        parameters = {
            "sigma_mode": "embedding_distortion_search",
            "sigma_alpha": best["alpha"],
            "sigma_base_distance": best["base_distance"],
            "sigma": best["sigma"],
            "gamma": best["gamma"],
            "embedding_distortion": best["distortion"],
            "n_neighbors": n_neighbors,
        }
    else:
        timings: dict[str, float] = {}
        t0 = perf_counter()
        W_sparse = knn_rbf_affinity_sparse(X, n_neighbors=n_neighbors, gamma=gamma)
        timings["graph_seconds"] = perf_counter() - t0
        t1 = perf_counter()
        vals, Y, affinity, mem = _spectral_embedding_from_sparse_affinity(W_sparse, n_clusters, dense_threshold)
        timings["eigensolver_seconds"] = perf_counter() - t1
        memory.update(mem)
        t2 = perf_counter()
        labels = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit_predict(Y)
        timings["kmeans_seconds"] = perf_counter() - t2
        timings["total_seconds"] = timings["graph_seconds"] + timings["eigensolver_seconds"] + timings["kmeans_seconds"]
        parameters = {"sigma_mode": "fixed_gamma", "gamma": gamma, "n_neighbors": n_neighbors}

    memory["embedding"] = int(Y.nbytes)

    return SpectralResult(
        labels=labels,
        embedding=Y,
        affinity=affinity,
        eigenvalues=vals,
        timings=timings,
        memory_bytes=memory,
        parameters=parameters,
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
        parameters={"affinity": "precomputed"},
    )


def dense_rbf_affinity(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Fully connected RBF affinity. Intended for moderate n only."""
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    if gamma is None:
        vals = sq[sq > 1e-12]
        sigma2 = float(np.median(vals)) if vals.size else 1.0
        gamma = 1.0 / max(sigma2, 1e-12)
    W = np.exp(-gamma * sq)
    np.fill_diagonal(W, 0.0)
    return W


def dense_rbf_affinity_sigma(X: np.ndarray, sigma: float) -> np.ndarray:
    """Fully connected RBF affinity using exp(-d^2 / (2 sigma^2))."""
    return dense_rbf_affinity(X, gamma=gamma_from_sigma(sigma))


def self_tuning_affinity(
    X: np.ndarray,
    local_neighbors: int = 7,
    mode: str = "nth",
    n_neighbors: int | None = None,
) -> np.ndarray | csr_matrix:
    """Local-scale affinity from Zelnik-Manor/Perona, optionally sparsified.

    mode="nth" sets sigma_i to the distance to the local_neighbors-th neighbor.
    mode="mean" sets sigma_i to the mean distance over those neighbors.
    """
    n = X.shape[0]
    p = min(max(local_neighbors, 1), n - 1)
    nn = NearestNeighbors(n_neighbors=p + 1, metric="minkowski", p=2).fit(X)
    dists, inds = nn.kneighbors(X)
    local_dists = dists[:, 1:]
    if mode == "nth":
        sigma = local_dists[:, -1]
    elif mode == "mean":
        sigma = local_dists.mean(axis=1)
    else:
        raise ValueError(f"unknown local scale mode: {mode}")
    sigma = np.maximum(sigma, 1e-12)

    if n_neighbors is None:
        D = pairwise_distances(X)
        denom = np.outer(sigma, sigma)
        W = np.exp(-np.square(D) / np.maximum(denom, 1e-12))
        np.fill_diagonal(W, 0.0)
        return W

    m = min(max(n_neighbors, 1), n - 1)
    rows = np.repeat(np.arange(n), m)
    cols = inds[:, 1:m + 1].ravel()
    vals = dists[:, 1:m + 1].ravel()
    denom = sigma[rows] * sigma[cols]
    weights = np.exp(-np.square(vals) / np.maximum(denom, 1e-12))
    W = csr_matrix((weights, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def mutual_knn_rbf_affinity_sparse(
    X: np.ndarray,
    n_neighbors: int = 10,
    gamma: float | None = None,
) -> csr_matrix:
    """Build a sparse mutual-kNN RBF graph: keep only edges i<->j that appear in both directed kNN lists."""
    G = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="distance",
        include_self=False,
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
    M = W.minimum(W.T).tocsr()
    M.setdiag(0.0)
    M.eliminate_zeros()
    return M


def spectral_from_affinity(
    W: np.ndarray | csr_matrix,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 20,
) -> SpectralResult:
    """Run the NJW spectral stage from a precomputed affinity matrix."""
    return spectral_on_graph(W, n_clusters=n_clusters, random_state=random_state, n_init=n_init)


def first_laplacian_eigenvalues(W: np.ndarray | csr_matrix, n_eigs: int = 12) -> np.ndarray:
    """Return the smallest eigenvalues of the symmetric normalized Laplacian."""
    if issparse(W):
        L = normalized_laplacian_sparse(W.tocsr())
        k = min(n_eigs, W.shape[0] - 2)
        vals, _ = eigsh(L, k=k, which="SM")
        return np.sort(vals)
    L = normalized_laplacian_dense(np.asarray(W))
    vals = eigh(L, eigvals_only=True)
    return vals[:n_eigs]
