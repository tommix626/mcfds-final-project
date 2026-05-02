from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

from .clustering import (
    dense_rbf_affinity,
    dense_rbf_affinity_sigma,
    first_laplacian_eigenvalues,
    gamma_from_sigma,
    kmeans_baseline,
    knn_rbf_affinity_sparse,
    knn_rbf_affinity_sigma_sparse,
    median_knn_distance,
    median_pairwise_distance,
    mutual_knn_rbf_affinity_sparse,
    ng_jordan_weiss,
    spectral_from_affinity,
    spectral_on_graph,
    self_tuning_affinity,
)
from .data import (
    digits_pca_2d,
    karate_graph,
    load_all_digits,
    load_digits_subset,
    load_iris_data,
    make_failure_datasets,
    make_success_datasets,
    stochastic_block_model_graph,
)
from .metrics import summarize
from .plots import (
    bar_comparison,
    eigenvalue_plot,
    groundtruth_comparison,
    heatmap,
    line_noise,
    line_scaling,
    multi_panel_scatter,
    scatter_clusters,
    sensitivity_scatter_cols,
    sensitivity_scatter_rows,
    sensitivity_slices,
    stacked_runtime,
)
from .utils import figures_dir, results_dir, save_json

SUCCESS_NEIGHBORS = {
    "two_moons": 12,
    "circles": 14,
    "blobs": 12,
    "high_dim_moons": 16,
    "high_dim_blobs": 18,
}

FAILURE_NEIGHBORS = {
    "overlapping_blobs_failure": 12,
    "bridge_failure": 14,
}

DIGITS_SPECS = [
    ("digits_01", (0, 1), 2, 3),
    ("digits_17", (1, 7), 2, 3),
    ("digits_0136", (0, 1, 3, 6), 4, 4),
    ("digits_all", tuple(range(10)), 10, 6),
]


def _record_rows(dataset: str, family: str, expectation: str, method: str, y_true: np.ndarray, y_pred: np.ndarray, extras: dict | None = None) -> dict:
    row = {
        "dataset": dataset,
        "family": family,
        "expectation": expectation,
        "method": method,
        **summarize(y_true, y_pred),
    }
    if extras:
        row.update(extras)
    return row


def _save_results(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(results_dir() / name, index=False)
    return df


def run_success_and_failure(random_state: int = 0) -> pd.DataFrame:
    figs = figures_dir()
    rows: list[dict] = []

    success = make_success_datasets(random_state=random_state)
    failure = make_failure_datasets(random_state=random_state)

    for family, datasets, neighbor_map in [
        ("success", success, SUCCESS_NEIGHBORS),
        ("stress", failure, FAILURE_NEIGHBORS),
    ]:
        for name, (X, y, k, expectation) in datasets.items():
            X_plot = X[:, :2] if X.shape[1] > 2 else X
            spec = ng_jordan_weiss(X, n_clusters=k, n_neighbors=neighbor_map[name], random_state=random_state)
            km = kmeans_baseline(X, n_clusters=k, random_state=random_state)
            rows.append(
                _record_rows(
                    name,
                    family,
                    expectation,
                    "spectral",
                    y,
                    spec.labels,
                    {**spec.timings, **{f"mem_{kk}": vv for kk, vv in spec.memory_bytes.items()}},
                )
            )
            rows.append(
                _record_rows(
                    name,
                    family,
                    expectation,
                    "kmeans",
                    y,
                    km.labels,
                    km.timings,
                )
            )
            xlab, ylab = ("x", "y") if X_plot.shape[1] == 2 else ("PC1", "PC2")
            scatter_clusters(figs / f"{name}_spectral.png", X_plot, spec.labels, f"{name}: spectral clustering", xlab, ylab)
            scatter_clusters(figs / f"{name}_kmeans.png", X_plot, km.labels, f"{name}: k-means", xlab, ylab)

    df = pd.DataFrame(rows)
    return _save_results(df, "synthetic_metrics.csv")


def run_real_data(random_state: int = 0) -> pd.DataFrame:
    rows: list[dict] = []

    X_iris, y_iris = load_iris_data()
    spec_iris = ng_jordan_weiss(X_iris, n_clusters=3, n_neighbors=12, random_state=random_state)
    km_iris = kmeans_baseline(X_iris, n_clusters=3, random_state=random_state)
    rows.append(_record_rows("iris", "real", "Spectral should be competitive; gains may be modest on nearly convex classes.", "spectral", y_iris, spec_iris.labels, spec_iris.timings))
    rows.append(_record_rows("iris", "real", "Spectral should be competitive; gains may be modest on nearly convex classes.", "kmeans", y_iris, km_iris.labels, km_iris.timings))

    X2, y2 = digits_pca_2d(classes=(0, 1, 3, 6), random_state=random_state)
    spec2 = ng_jordan_weiss(X2, n_clusters=4, n_neighbors=10, random_state=random_state)
    km2 = kmeans_baseline(X2, n_clusters=4, random_state=random_state)
    scatter_clusters(figures_dir() / "digits_2d_spectral.png", X2, spec2.labels, "Digits {0,1,3,6} PCA(2): spectral", "PC1", "PC2")
    scatter_clusters(figures_dir() / "digits_2d_kmeans.png", X2, km2.labels, "Digits {0,1,3,6} PCA(2): k-means", "PC1", "PC2")

    X_all, y_all = load_all_digits()
    for name, classes, k, neighbors in DIGITS_SPECS:
        if name == "digits_all":
            Xd, yd = X_all, y_all
        else:
            Xd, yd = load_digits_subset(classes=classes, random_state=random_state)
        spec = ng_jordan_weiss(Xd, n_clusters=k, n_neighbors=neighbors, random_state=random_state)
        km = kmeans_baseline(Xd, n_clusters=k, random_state=random_state)
        expectation = "Digits form semantically meaningful but imperfect clusters; graph construction matters strongly here."
        rows.append(_record_rows(name, "digits", expectation, "spectral", yd, spec.labels, spec.timings))
        rows.append(_record_rows(name, "digits", expectation, "kmeans", yd, km.labels, km.timings))

    Gk, yk = karate_graph()
    Ak = nx.to_numpy_array(Gk, weight="weight")
    spec_g = spectral_on_graph(Ak, n_clusters=2, random_state=random_state)
    km_g = kmeans_baseline(Ak, n_clusters=2, random_state=random_state)
    rows.append(_record_rows("karate_graph", "graph", "Spectral should recover the post-split factions more naturally than adjacency-row k-means.", "spectral", yk, spec_g.labels, spec_g.timings))
    rows.append(_record_rows("karate_graph", "graph", "Spectral should recover the post-split factions more naturally than adjacency-row k-means.", "kmeans_on_adjacency", yk, km_g.labels, km_g.timings))

    import matplotlib.pyplot as plt

    pos = nx.spring_layout(Gk, seed=random_state)
    plt.figure(figsize=(5.4, 4.4))
    nx.draw_networkx(Gk, pos=pos, node_color=spec_g.labels, with_labels=False, node_size=160)
    plt.title("Karate club: spectral clustering on graph")
    plt.tight_layout()
    plt.savefig(figures_dir() / "karate_spectral.png", dpi=220)
    plt.close()

    Gs, ys = stochastic_block_model_graph(random_state=random_state)
    As = nx.to_numpy_array(Gs)
    spec_sbm = spectral_on_graph(As, n_clusters=3, random_state=random_state)
    km_sbm = kmeans_baseline(As, n_clusters=3, random_state=random_state)
    rows.append(_record_rows("sbm_graph", "graph", "On a planted-community graph, spectral should strongly recover communities.", "spectral", ys, spec_sbm.labels, spec_sbm.timings))
    rows.append(_record_rows("sbm_graph", "graph", "On a planted-community graph, spectral should strongly recover communities.", "kmeans_on_adjacency", ys, km_sbm.labels, km_sbm.timings))

    pos2 = nx.spring_layout(Gs, seed=random_state)
    plt.figure(figsize=(5.4, 4.4))
    nx.draw_networkx(Gs, pos=pos2, node_color=spec_sbm.labels, with_labels=False, node_size=55)
    plt.title("Stochastic block model: spectral clustering")
    plt.tight_layout()
    plt.savefig(figures_dir() / "sbm_spectral.png", dpi=220)
    plt.close()

    df = pd.DataFrame(rows)
    _save_results(df[df["dataset"].str.startswith("digits")], "digits_metrics.csv")
    _save_results(df[df["dataset"] == "iris"], "iris_metrics.csv")
    _save_results(df[df["family"] == "graph"], "graph_metrics.csv")
    return _save_results(df, "real_metrics.csv")


def run_noise_experiment(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    X, y = datasets.make_moons(n_samples=550, noise=0.06, random_state=random_state)
    sigmas = [0.0, 0.08, 0.16, 0.24, 0.32, 0.40]
    spec_scores = []
    km_scores = []
    for sigma in sigmas:
        rs = np.random.default_rng(random_state)
        Xn = X + rs.normal(scale=sigma, size=X.shape)
        spec = ng_jordan_weiss(Xn, n_clusters=2, n_neighbors=12, random_state=random_state)
        km = kmeans_baseline(Xn, n_clusters=2, random_state=random_state)
        spec_scores.append(summarize(y, spec.labels)["ari"])
        km_scores.append(summarize(y, km.labels)["ari"])
    line_noise(figures_dir() / "noise_moons_ari.png", sigmas, spec_scores, km_scores, "Two moons under additive Gaussian noise", "ARI")
    df = pd.DataFrame({"sigma": sigmas, "spectral_ari": spec_scores, "kmeans_ari": km_scores})
    return _save_results(df, "noise_metrics.csv")


def run_scaling_benchmark(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    ns = [150, 300, 600, 900]
    rows: list[dict] = []
    for n in ns:
        X, y = datasets.make_moons(n_samples=n, noise=0.07, random_state=random_state)
        spec = ng_jordan_weiss(X, n_clusters=2, n_neighbors=12, random_state=random_state)
        km = kmeans_baseline(X, n_clusters=2, random_state=random_state)
        rows.append({
            "n": n,
            "method": "spectral",
            "graph_seconds": spec.timings["graph_seconds"],
            "eigensolver_seconds": spec.timings["eigensolver_seconds"],
            "post_kmeans_seconds": spec.timings["kmeans_seconds"],
            "total_seconds": spec.timings["total_seconds"],
            "affinity_bytes": spec.memory_bytes.get("affinity_sparse", spec.memory_bytes.get("affinity_dense", 0)),
            "embedding_bytes": spec.memory_bytes["embedding"],
            "ari": summarize(y, spec.labels)["ari"],
        })
        rows.append({
            "n": n,
            "method": "kmeans",
            "graph_seconds": 0.0,
            "eigensolver_seconds": 0.0,
            "post_kmeans_seconds": km.timings["kmeans_seconds"],
            "total_seconds": km.timings["kmeans_seconds"],
            "affinity_bytes": 0,
            "embedding_bytes": 0,
            "ari": summarize(y, km.labels)["ari"],
        })

    df = pd.DataFrame(rows)
    sdf = df[df["method"] == "spectral"].copy()
    line_scaling(
        figures_dir() / "scaling_runtime.png",
        sdf["n"].tolist(),
        {
            "graph build": sdf["graph_seconds"].tolist(),
            "eigensolver": sdf["eigensolver_seconds"].tolist(),
            "spectral total": sdf["total_seconds"].tolist(),
        },
        "Spectral clustering runtime versus sample size",
        "Number of samples n",
        "Seconds",
    )
    line_scaling(
        figures_dir() / "scaling_memory.png",
        sdf["n"].tolist(),
        {
            "affinity bytes": (sdf["affinity_bytes"] / 1024).tolist(),
            "embedding bytes": (sdf["embedding_bytes"] / 1024).tolist(),
        },
        "Approximate memory footprint versus sample size",
        "Number of samples n",
        "KiB",
    )
    return _save_results(df, "scaling_metrics.csv")


def run_summary_figures() -> None:
    synthetic = pd.read_csv(results_dir() / "synthetic_metrics.csv")
    real_df = pd.read_csv(results_dir() / "real_metrics.csv")
    digits = pd.read_csv(results_dir() / "digits_metrics.csv")

    syn_spec = synthetic[synthetic["method"] == "spectral"]
    syn_km = synthetic[synthetic["method"] == "kmeans"]
    categories = syn_spec["dataset"].tolist()
    bar_comparison(
        figures_dir() / "synthetic_ari_bar.png",
        categories,
        syn_spec["ari"].tolist(),
        syn_km["ari"].tolist(),
        "Spectral",
        "k-means",
        "ARI across synthetic datasets",
        "ARI",
    )

    dig_spec = digits[digits["method"] == "spectral"]
    dig_km = digits[digits["method"] == "kmeans"]
    bar_comparison(
        figures_dir() / "digits_purity_bar.png",
        dig_spec["dataset"].tolist(),
        dig_spec["purity"].tolist(),
        dig_km["purity"].tolist(),
        "Spectral",
        "k-means",
        "Purity on digits subsets",
        "Purity",
    )

    best = {}
    for name, group in pd.concat([synthetic, real_df], ignore_index=True).groupby("dataset"):
        best_row = group.sort_values(["ari", "purity", "accuracy"], ascending=False).iloc[0]
        best[name] = {
            "best_method": best_row["method"],
            "ari": float(best_row["ari"]),
            "purity": float(best_row["purity"]),
            "accuracy": float(best_row["accuracy"]),
        }
    save_json(best, results_dir() / "best_methods.json")


def run_parameter_sensitivity(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    noises = [0.02, 0.08, 0.14, 0.20, 0.28, 0.36]
    neighbors = [3, 5, 8, 12, 18, 28]

    # Representative indices for scatter grids
    ROW_NOISE_IDXS = [0, 4, 5]      # noise=0.02, 0.28, 0.36
    COL_NEIGHBOR_IDXS = [0, 2, 3, 5]  # k=3, 8, 12, 28
    COL_FIXED_NEIGHBOR_IDX = 3       # k=12 held fixed for column sweep

    scatter_cache: dict = {}
    rows: list[dict] = []
    spectral_mat = np.zeros((len(noises), len(neighbors)))
    kmeans_col = np.zeros(len(noises))
    for i, noise in enumerate(noises):
        X, y = datasets.make_moons(n_samples=500, noise=noise, random_state=random_state)
        km = kmeans_baseline(X, n_clusters=2, random_state=random_state)
        km_ari = summarize(y, km.labels)["ari"]
        kmeans_col[i] = km_ari
        for j, nn in enumerate(neighbors):
            try:
                spec = ng_jordan_weiss(X, n_clusters=2, n_neighbors=nn, random_state=random_state)
                ari = summarize(y, spec.labels)["ari"]
            except Exception:
                spec = ng_jordan_weiss(X, n_clusters=2, n_neighbors=nn, random_state=random_state, dense_threshold=X.shape[0] + 1)
                ari = summarize(y, spec.labels)["ari"]
            spectral_mat[i, j] = ari
            rows.append({"dataset": "two_moons", "noise": noise, "n_neighbors": nn, "spectral_ari": ari, "kmeans_ari_at_same_noise": km_ari})

            # Cache scatter data for representative configurations
            need_for_rows = (i in ROW_NOISE_IDXS and j in COL_NEIGHBOR_IDXS)
            need_for_cols = (i in range(len(noises)) and j == COL_FIXED_NEIGHBOR_IDX)
            if need_for_rows or need_for_cols:
                scatter_cache[(i, j)] = (X.copy(), y.copy(), spec.labels.copy(), ari)

    full_mat = np.concatenate([spectral_mat, kmeans_col[:, None]], axis=1)
    heatmap(
        figures_dir() / "parameter_sensitivity_heatmap.png",
        full_mat,
        [str(x) for x in neighbors] + ["k-means"],
        [str(x) for x in noises],
        "Spectral clustering sensitivity on two moons",
        "Graph setting",
        "Noise standard deviation",
        "ARI",
        separator_before_last=True,
    )

    # Scatter grid: rows = noise levels, cols = kNN degrees
    sensitivity_scatter_rows(
        figures_dir() / "parameter_sensitivity_rows.png",
        scatter_cache,
        noises,
        neighbors,
        row_noise_idxs=ROW_NOISE_IDXS,
        col_neighbor_idxs=COL_NEIGHBOR_IDXS,
    )

    # Scatter strip: fixed k=12, all noise levels
    sensitivity_scatter_cols(
        figures_dir() / "parameter_sensitivity_cols.png",
        scatter_cache,
        noises,
        neighbors,
        all_noise_idxs=list(range(len(noises))),
        fixed_neighbor_idx=COL_FIXED_NEIGHBOR_IDX,
    )

    return _save_results(pd.DataFrame(rows), "parameter_sensitivity_metrics.csv")


def run_eigengap_study(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    specs: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
    X_m, y_m = datasets.make_moons(n_samples=500, noise=0.08, random_state=random_state)
    specs.append(("moons", X_m, y_m, 2, 12))
    X_c, y_c = datasets.make_circles(n_samples=500, factor=0.45, noise=0.06, random_state=random_state)
    specs.append(("circles", X_c, y_c, 2, 14))
    X_o, y_o = datasets.make_blobs(n_samples=500, centers=[[0, 0], [0.9, 0.9], [1.8, 1.8]], cluster_std=1.0, random_state=random_state)
    specs.append(("overlap", X_o, y_o, 3, 12))
    X_d, y_d = load_digits_subset(classes=(0, 1, 3, 6), random_state=random_state)
    specs.append(("digits_0136", X_d, y_d, 4, 4))
    Gs, ys = stochastic_block_model_graph(random_state=random_state)
    As = nx.to_numpy_array(Gs)
    eigs_dict: dict[str, list[float]] = {}
    rows: list[dict] = []
    for name, X, y, k, nn in specs:
        W = knn_rbf_affinity_sparse(X, n_neighbors=nn)
        vals = first_laplacian_eigenvalues(W, n_eigs=12)
        spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
        metrics = summarize(y, spec.labels)
        eigs_dict[name] = vals.tolist()
        gap_after_k = float(vals[k] - vals[k - 1]) if len(vals) > k else np.nan
        for idx, val in enumerate(vals, start=1):
            rows.append({"dataset": name, "index": idx, "eigenvalue": float(val), "k_true": k, "gap_after_k": gap_after_k, "ari": metrics["ari"]})
    vals_sbm = first_laplacian_eigenvalues(As, n_eigs=12)
    spec_sbm = spectral_from_affinity(As, n_clusters=3, random_state=random_state)
    metrics_sbm = summarize(ys, spec_sbm.labels)
    eigs_dict["sbm"] = vals_sbm.tolist()
    gap_sbm = float(vals_sbm[3] - vals_sbm[2]) if len(vals_sbm) > 3 else np.nan
    for idx, val in enumerate(vals_sbm, start=1):
        rows.append({"dataset": "sbm", "index": idx, "eigenvalue": float(val), "k_true": 3, "gap_after_k": gap_sbm, "ari": metrics_sbm["ari"]})
    k_true_dict = {name: k for name, X, y, k, nn in specs}
    k_true_dict["sbm"] = 3
    eigenvalue_plot(figures_dir() / "eigengap_study.png", eigs_dict, "Small Laplacian eigenvalues across datasets", k_true_dict=k_true_dict)
    return _save_results(pd.DataFrame(rows), "eigengap_metrics.csv")


def run_graph_construction_ablation(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    dataset_specs = []
    X_m, y_m = datasets.make_moons(n_samples=500, noise=0.10, random_state=random_state)
    dataset_specs.append(("moons", X_m, y_m, 2, 12))
    X_c, y_c = datasets.make_circles(n_samples=500, factor=0.45, noise=0.08, random_state=random_state)
    dataset_specs.append(("circles", X_c, y_c, 2, 14))
    X_d, y_d = load_digits_subset(classes=(0, 1, 3, 6), random_state=random_state)
    dataset_specs.append(("digits_0136", X_d, y_d, 4, 4))
    rows: list[dict] = []
    for name, X, y, k, nn in dataset_specs:
        graphs = {
            "knn_rbf": knn_rbf_affinity_sparse(X, n_neighbors=nn),
            "mutual_knn_rbf": mutual_knn_rbf_affinity_sparse(X, n_neighbors=nn),
            "dense_rbf": dense_rbf_affinity(X),
        }
        for graph_name, W in graphs.items():
            try:
                spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
                rows.append({"dataset": name, "graph": graph_name, **summarize(y, spec.labels)})
            except Exception as e:
                rows.append({"dataset": name, "graph": graph_name, "accuracy": np.nan, "ari": np.nan, "nmi": np.nan, "purity": np.nan, "error": str(e)})
    df = pd.DataFrame(rows)
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="dataset", columns="graph", values="ari")
    pivot = pivot[["dense_rbf", "knn_rbf", "mutual_knn_rbf"]]
    pivot.plot(kind="bar", figsize=(7.0, 4.4))
    plt.ylabel("ARI")
    plt.title("Graph construction ablation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir() / "graph_construction_ablation.png", dpi=220)
    plt.close()
    return _save_results(df, "graph_construction_ablation.csv")


def run_failure_taxonomy(random_state: int = 0) -> pd.DataFrame:
    from sklearn import datasets

    rows: list[dict] = []
    panels = []

    X1, y1 = datasets.make_moons(n_samples=220, noise=0.08, random_state=random_state)
    spec1 = ng_jordan_weiss(X1, n_clusters=2, n_neighbors=2, random_state=random_state, dense_threshold=250)
    rows.append({"failure_mode": "too_few_neighbors_fragmentation", "dataset": "moons", "n_neighbors": 2, **summarize(y1, spec1.labels)})
    panels.append((X1, spec1.labels, "Too few neighbors"))

    rs = np.random.default_rng(random_state)
    left = rs.normal(loc=[-3.3, 0.0], scale=[0.42, 0.42], size=(220, 2))
    right = rs.normal(loc=[3.3, 0.0], scale=[0.42, 0.42], size=(220, 2))
    bridge_x = np.linspace(-2.8, 2.8, 180)
    bridge_y = rs.normal(loc=0.0, scale=0.06, size=bridge_x.shape[0])
    bridge = np.c_[bridge_x, bridge_y]
    X2 = np.vstack([left, right, bridge])
    y2 = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int), np.zeros(len(bridge), dtype=int)])
    spec2 = ng_jordan_weiss(X2, n_clusters=2, n_neighbors=16, random_state=random_state)
    rows.append({"failure_mode": "thin_long_bridge", "dataset": "bridge_long", "n_neighbors": 16, **summarize(y2, spec2.labels)})
    panels.append((X2, spec2.labels, "Long thin bridge"))

    dense = rs.normal(loc=[-2.2, 0.1], scale=0.20, size=(90, 2))
    diffuse = rs.normal(loc=[1.9, 0.0], scale=1.15, size=(170, 2))
    tiny = rs.normal(loc=[-0.1, 2.4], scale=0.23, size=(24, 2))
    X3 = np.vstack([dense, diffuse, tiny])
    y3 = np.concatenate([
        np.zeros(len(dense), dtype=int),
        np.ones(len(diffuse), dtype=int),
        np.full(len(tiny), 2, dtype=int),
    ])
    spec3 = ng_jordan_weiss(X3, n_clusters=3, n_neighbors=10, random_state=random_state, dense_threshold=400)
    rows.append({"failure_mode": "varying_density_fragility", "dataset": "density_fragility", "n_neighbors": 10, **summarize(y3, spec3.labels)})
    panels.append((X3, spec3.labels, "Varying density"))

    multi_panel_scatter(figures_dir() / "failure_taxonomy.png", panels, "Graph fragility archetypes", xlabel="x", ylabel="y")
    return _save_results(pd.DataFrame(rows), "failure_taxonomy_metrics.csv")


def run_failure_comparison_figures(random_state: int = 0) -> None:
    """Regenerate 3-panel ground-truth/k-means/spectral figures for each failure mode."""
    failure = make_failure_datasets(random_state=random_state)
    figs = figures_dir()
    name_to_outfile = {
        "overlapping_blobs_failure": "overlapping_groundtruth_kmeans_njw.png",
        "bridge_failure": "bridge_groundtruth_kmeans_njw.png",
    }
    title_map = {
        "overlapping_blobs_failure": "Overlapping blobs: ground truth vs. k-means vs. NJW spectral",
        "bridge_failure": "Bridge stress case: ground truth vs. k-means vs. NJW spectral",
    }
    for name, (X, y, k, _) in failure.items():
        spec = ng_jordan_weiss(X, n_clusters=k, n_neighbors=FAILURE_NEIGHBORS[name], random_state=random_state)
        km = kmeans_baseline(X, n_clusters=k, random_state=random_state)
        groundtruth_comparison(
            figs / name_to_outfile[name],
            X, y, km.labels, spec.labels,
            title=title_map[name],
        )


SIGMA_ALPHAS = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1.40, 2.00, 3.00, 5.00]
SPARSIFICATION_NEIGHBORS = [2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48]


def _affinity_bytes(W) -> int:
    if issparse(W):
        A = W.tocsr()
        return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    return int(np.asarray(W).nbytes)


def _embedding_distortion(Y: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lab in np.unique(labels):
        pts = Y[labels == lab]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0, keepdims=True)
        total += float(np.sum((pts - center) ** 2))
    return total


def _diagnostic_specs(random_state: int = 0):
    from sklearn import datasets

    success = make_success_datasets(random_state=random_state)
    specs = []
    for name, classes, k, neighbors in DIGITS_SPECS:
        if name == "digits_all":
            X, y = load_all_digits()
        else:
            X, y = load_digits_subset(classes=classes, random_state=random_state)
        specs.append((name, X, y, k, neighbors, "digits"))
    X_iris, y_iris = load_iris_data()
    specs.append(("iris", X_iris, y_iris, 3, 12, "real"))
    for name in ["two_moons", "circles", "blobs"]:
        X, y, k, _ = success[name]
        specs.append((name, X, y, k, SUCCESS_NEIGHBORS[name], "synthetic"))
    return specs


def _variable_density_data(random_state: int = 0):
    rs = np.random.default_rng(random_state)
    dense = rs.normal(loc=[-2.0, 0.0], scale=[0.22, 0.22], size=(180, 2))
    diffuse = rs.normal(loc=[1.2, 0.0], scale=[0.95, 0.95], size=(240, 2))
    small = rs.normal(loc=[0.0, 2.25], scale=[0.28, 0.28], size=(80, 2))
    X = np.vstack([dense, diffuse, small])
    y = np.concatenate([
        np.zeros(len(dense), dtype=int),
        np.ones(len(diffuse), dtype=int),
        np.full(len(small), 2, dtype=int),
    ])
    return X, y, 3, 12


def _run_affinity_configuration(W, y, k, random_state: int, graph_seconds: float, extra: dict) -> dict:
    row = dict(extra)
    row["graph_seconds"] = graph_seconds
    row["affinity_bytes"] = _affinity_bytes(W)
    row["nnz_affinity"] = int(W.nnz if issparse(W) else np.count_nonzero(W))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
        row.update(summarize(y, spec.labels))
        row["eigensolver_seconds"] = spec.timings.get("eigensolver_seconds", np.nan)
        row["post_kmeans_seconds"] = spec.timings.get("kmeans_seconds", np.nan)
        row["total_seconds"] = graph_seconds + spec.timings.get("total_seconds", 0.0)
        row["embedding_distortion"] = _embedding_distortion(spec.embedding, spec.labels)
        row["error"] = ""
    except Exception as exc:
        row.update({"accuracy": np.nan, "ari": np.nan, "nmi": np.nan, "purity": np.nan})
        row["eigensolver_seconds"] = np.nan
        row["post_kmeans_seconds"] = np.nan
        row["total_seconds"] = np.nan
        row["embedding_distortion"] = np.nan
        row["error"] = str(exc)
    return row


def _build_sigma_affinity(X: np.ndarray, graph: str, n_neighbors: int, alpha: float):
    if graph == "dense_rbf":
        base = median_pairwise_distance(X)
        sigma = alpha * base
        t0 = perf_counter()
        W = dense_rbf_affinity_sigma(X, sigma=sigma)
        graph_seconds = perf_counter() - t0
    elif graph == "knn_rbf":
        base = median_knn_distance(X, n_neighbors=n_neighbors)
        sigma = alpha * base
        t0 = perf_counter()
        W = knn_rbf_affinity_sigma_sparse(X, n_neighbors=n_neighbors, sigma=sigma)
        graph_seconds = perf_counter() - t0
    else:
        raise ValueError(f"unknown graph type: {graph}")
    return W, sigma, base, graph_seconds


def run_sigma_sweep(random_state: int = 0) -> pd.DataFrame:
    rows = []
    baselines = []
    for dataset, X, y, k, nn, family in _diagnostic_specs(random_state=random_state):
        km = kmeans_baseline(X, n_clusters=k, random_state=random_state)
        km_metrics = summarize(y, km.labels)
        baselines.append({
            "dataset": dataset,
            "kmeans_accuracy": km_metrics["accuracy"],
            "kmeans_ari": km_metrics["ari"],
            "kmeans_nmi": km_metrics["nmi"],
            "kmeans_purity": km_metrics["purity"],
        })
        for graph in ["dense_rbf", "knn_rbf"]:
            for alpha in SIGMA_ALPHAS:
                W, sigma, base, graph_seconds = _build_sigma_affinity(X, graph, nn, alpha)
                rows.append(_run_affinity_configuration(
                    W,
                    y,
                    k,
                    random_state,
                    graph_seconds,
                    {
                        "dataset": dataset,
                        "family": family,
                        "graph": graph,
                        "alpha": alpha,
                        "base_distance": base,
                        "sigma": sigma,
                        "gamma": gamma_from_sigma(sigma),
                        "n_neighbors": nn if graph == "knn_rbf" else np.nan,
                    },
                ))

    df = _save_results(pd.DataFrame(rows), "sigma_sweep_metrics.csv")
    baseline_df = pd.DataFrame(baselines)
    best = (
        df[df["error"].fillna("") == ""]
        .sort_values(["dataset", "graph", "ari", "purity", "accuracy"], ascending=[True, True, False, False, False])
        .groupby(["dataset", "graph"], as_index=False)
        .head(1)
        .merge(baseline_df, on="dataset", how="left")
    )
    _save_results(best, "sigma_sweep_best_by_dataset.csv")
    _plot_sigma_sweep(df, best, baseline_df)
    run_sigma_eigengap_sweep(random_state=random_state)
    return df


def _plot_sigma_sweep(df: pd.DataFrame, best: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    digits = df[df["dataset"].str.startswith("digits")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True, sharey=True)
    for ax, dataset in zip(axes.ravel(), ["digits_01", "digits_17", "digits_0136", "digits_all"]):
        part = digits[(digits["dataset"] == dataset) & (digits["error"].fillna("") == "")]
        for graph, label in [("dense_rbf", "dense RBF"), ("knn_rbf", "kNN-RBF")]:
            sub = part[part["graph"] == graph].sort_values("alpha")
            ax.plot(sub["alpha"], sub["ari"], marker="o", label=label)
        base = baseline_df[baseline_df["dataset"] == dataset]
        if not base.empty:
            ax.axhline(float(base.iloc[0]["kmeans_ari"]), color="black", linestyle="--", linewidth=1.2, label="k-means")
        ax.set_title(dataset)
        ax.set_xlabel(r"$\alpha$ in $\sigma=\alpha r_{\mathrm{med}}$")
        ax.set_ylabel("ARI")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Digit sensitivity to RBF scale")
    fig.tight_layout()
    fig.savefig(figures_dir() / "digits_sigma_sweep_ari.png", dpi=220)
    plt.close(fig)

    pivot = best[best["dataset"].str.startswith("digits")].pivot(index="dataset", columns="graph", values="ari")
    labels = [f"{v:.2f}" for v in best[best["dataset"].str.startswith("digits")]["alpha"].tolist()]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Best digit ARI over sigma sweep")
    for i, ds in enumerate(pivot.index):
        for j, graph in enumerate(pivot.columns):
            row = best[(best["dataset"] == ds) & (best["graph"] == graph)].iloc[0]
            ax.text(j, i, f"ARI={row['ari']:.2f}\nα={row['alpha']:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best ARI")
    fig.tight_layout()
    fig.savefig(figures_dir() / "digits_sigma_sweep_summary.png", dpi=220)
    plt.close(fig)


def run_adaptive_sigma(random_state: int = 0) -> pd.DataFrame:
    specs = [("variable_density", *_variable_density_data(random_state=random_state), "synthetic")]
    for dataset, X, y, k, nn, family in _diagnostic_specs(random_state=random_state):
        if dataset.startswith("digits") or dataset in {"two_moons", "circles", "blobs"}:
            specs.append((dataset, X, y, k, nn, family))

    rows = []
    for dataset, X, y, k, nn, family in specs:
        base = median_knn_distance(X, n_neighbors=nn)
        sigma = 0.70 * base
        methods = [
            ("fixed_global_knn", lambda: knn_rbf_affinity_sigma_sparse(X, n_neighbors=nn, sigma=sigma)),
            ("local_nth_dense", lambda: self_tuning_affinity(X, local_neighbors=7, mode="nth")),
            ("local_mean_dense", lambda: self_tuning_affinity(X, local_neighbors=7, mode="mean")),
        ]
        for method, build in methods:
            t0 = perf_counter()
            W = build()
            graph_seconds = perf_counter() - t0
            rows.append(_run_affinity_configuration(
                W,
                y,
                k,
                random_state,
                graph_seconds,
                {
                    "dataset": dataset,
                    "family": family,
                    "method": method,
                    "local_neighbors": 7 if method.startswith("local") else np.nan,
                    "n_neighbors": nn if method == "fixed_global_knn" else np.nan,
                    "sigma": sigma if method == "fixed_global_knn" else np.nan,
                },
            ))
    df = _save_results(pd.DataFrame(rows), "adaptive_sigma_metrics.csv")
    _plot_adaptive_sigma(df, random_state=random_state)
    return df


def _plot_adaptive_sigma(df: pd.DataFrame, random_state: int = 0) -> None:
    import matplotlib.pyplot as plt

    X, y, k, nn = _variable_density_data(random_state=random_state)
    panels = [("Ground truth", y)]
    for method in ["fixed_global_knn", "local_nth_dense", "local_mean_dense"]:
        if method == "fixed_global_knn":
            sigma = 0.70 * median_knn_distance(X, n_neighbors=nn)
            W = knn_rbf_affinity_sigma_sparse(X, n_neighbors=nn, sigma=sigma)
        else:
            mode = "nth" if method == "local_nth_dense" else "mean"
            W = self_tuning_affinity(X, local_neighbors=7, mode=mode)
        spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
        panels.append((method.replace("_", " "), spec.labels))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6))
    for ax, (title, labels) in zip(axes, panels):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8, linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Adaptive sigma on variable-density data")
    fig.tight_layout()
    fig.savefig(figures_dir() / "adaptive_sigma_variable_density.png", dpi=220)
    plt.close(fig)

    plot_df = df[df["error"].fillna("") == ""].copy()
    order = ["variable_density", "digits_0136", "digits_all", "two_moons", "circles", "blobs"]
    plot_df = plot_df[plot_df["dataset"].isin(order)]
    pivot = plot_df.pivot(index="dataset", columns="method", values="ari").reindex(order)
    pivot.plot(kind="bar", figsize=(9.0, 4.8))
    plt.ylabel("ARI")
    plt.title("Fixed global sigma versus local adaptive sigma")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir() / "adaptive_sigma_metrics.png", dpi=220)
    plt.close()


def run_sigma_eigengap_sweep(random_state: int = 0) -> pd.DataFrame:
    specs = [("variable_density", *_variable_density_data(random_state=random_state), "synthetic")]
    for dataset, X, y, k, nn, family in _diagnostic_specs(random_state=random_state):
        if dataset.startswith("digits") or dataset in {"two_moons", "circles"}:
            specs.append((dataset, X, y, k, nn, family))
    rows = []
    for dataset, X, y, k, nn, family in specs:
        for graph in ["dense_rbf", "knn_rbf"]:
            for alpha in SIGMA_ALPHAS:
                W, sigma, base, graph_seconds = _build_sigma_affinity(X, graph, nn, alpha)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        vals = first_laplacian_eigenvalues(W, n_eigs=12)
                        spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
                    metrics = summarize(y, spec.labels)
                    gaps = np.diff(vals)
                    max_check = min(len(gaps), 10)
                    pred_k = int(np.argmax(gaps[:max_check]) + 1) if max_check > 0 else np.nan
                    true_gap = float(vals[k] - vals[k - 1]) if len(vals) > k else np.nan
                    error = ""
                except Exception as exc:
                    vals = np.full(12, np.nan)
                    metrics = {"accuracy": np.nan, "ari": np.nan, "nmi": np.nan, "purity": np.nan}
                    pred_k = np.nan
                    true_gap = np.nan
                    error = str(exc)
                for idx, val in enumerate(vals, start=1):
                    rows.append({
                        "dataset": dataset,
                        "family": family,
                        "graph": graph,
                        "alpha": alpha,
                        "base_distance": base,
                        "sigma": sigma,
                        "gamma": gamma_from_sigma(sigma),
                        "eigen_index": idx,
                        "eigenvalue_lsym": float(val),
                        "k_true": k,
                        "true_k_gap": true_gap,
                        "predicted_k_largest_gap": pred_k,
                        "graph_seconds": graph_seconds,
                        **metrics,
                        "error": error,
                    })
    df = _save_results(pd.DataFrame(rows), "sigma_eigengap_sweep.csv")
    _plot_sigma_eigengap_sweep(df)
    return df


def _plot_sigma_eigengap_sweep(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    summary = df[df["eigen_index"] == 1].copy()
    digits = summary[summary["dataset"].str.startswith("digits")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for ax, dataset in zip(axes.ravel(), ["digits_01", "digits_17", "digits_0136", "digits_all"]):
        part = digits[(digits["dataset"] == dataset) & (digits["error"].fillna("") == "")]
        for graph, label in [("dense_rbf", "dense RBF"), ("knn_rbf", "kNN-RBF")]:
            sub = part[part["graph"] == graph].sort_values("alpha")
            ax.plot(sub["alpha"], sub["true_k_gap"], marker="o", label=f"{label} true-k gap")
        ax2 = ax.twinx()
        sub = part[part["graph"] == "knn_rbf"].sort_values("alpha")
        ax2.plot(sub["alpha"], sub["predicted_k_largest_gap"], color="gray", linestyle="--", marker="x", label="kNN predicted k")
        ax.set_title(dataset)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$ in $\sigma=\alpha r_{\mathrm{med}}$")
        ax.set_ylabel("True-k eigengap")
        ax2.set_ylabel("Predicted k")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Digit eigengap sensitivity to RBF scale")
    fig.tight_layout()
    fig.savefig(figures_dir() / "digits_sigma_eigengap_sweep.png", dpi=220)
    plt.close(fig)


def run_sparsification_tradeoff(random_state: int = 0) -> pd.DataFrame:
    rows = []
    for dataset, X, y, k, nn_default, family in _diagnostic_specs(random_state=random_state):
        if dataset not in {"digits_0136", "digits_all", "two_moons", "circles", "blobs"}:
            continue
        dense_sigma = 0.70 * median_pairwise_distance(X)
        t0 = perf_counter()
        W_dense = dense_rbf_affinity_sigma(X, sigma=dense_sigma)
        dense_graph_seconds = perf_counter() - t0
        rows.append(_run_affinity_configuration(
            W_dense,
            y,
            k,
            random_state,
            dense_graph_seconds,
            {
                "dataset": dataset,
                "family": family,
                "graph": "dense_rbf_reference",
                "n_neighbors": np.nan,
                "alpha": 0.70,
                "sigma": dense_sigma,
                "gamma": gamma_from_sigma(dense_sigma),
            },
        ))
        for nn in SPARSIFICATION_NEIGHBORS:
            if nn >= X.shape[0]:
                continue
            base = median_knn_distance(X, n_neighbors=nn)
            sigma = 0.70 * base
            t0 = perf_counter()
            W = knn_rbf_affinity_sigma_sparse(X, n_neighbors=nn, sigma=sigma)
            graph_seconds = perf_counter() - t0
            rows.append(_run_affinity_configuration(
                W,
                y,
                k,
                random_state,
                graph_seconds,
                {
                    "dataset": dataset,
                    "family": family,
                    "graph": "knn_rbf",
                    "n_neighbors": nn,
                    "alpha": 0.70,
                    "sigma": sigma,
                    "gamma": gamma_from_sigma(sigma),
                },
            ))
    df = _save_results(pd.DataFrame(rows), "sparsification_tradeoff_metrics.csv")
    _plot_sparsification_tradeoff(df)
    return df


def _plot_sparsification_tradeoff(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_df = df[df["error"].fillna("") == ""]
    for xcol, xlabel, outfile in [
        ("total_seconds", "Total seconds", "sparsification_pareto_runtime.png"),
        ("nnz_affinity", "Number of nonzero affinity entries", "sparsification_pareto_edges.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for dataset in ["digits_0136", "digits_all", "two_moons", "circles", "blobs"]:
            sub = plot_df[(plot_df["dataset"] == dataset) & (plot_df["graph"] == "knn_rbf")].sort_values(xcol)
            if not sub.empty:
                ax.plot(sub[xcol], sub["ari"], marker="o", label=f"{dataset} kNN")
            dense = plot_df[(plot_df["dataset"] == dataset) & (plot_df["graph"] == "dense_rbf_reference")]
            if not dense.empty:
                ax.scatter(dense[xcol], dense["ari"], marker="X", s=70, label=f"{dataset} dense")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ARI")
        ax.set_title("Sparsification tradeoff")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir() / outfile, dpi=220)
        plt.close(fig)


def run_bottleneck_breakdown(random_state: int = 0) -> pd.DataFrame:
    df = pd.read_csv(results_dir() / "scaling_metrics.csv")
    sdf = df[df["method"] == "spectral"].copy()
    kdf = df[df["method"] == "kmeans"].copy()
    stacked_runtime(figures_dir() / "runtime_bottleneck_breakdown.png", sdf, "Spectral pipeline bottleneck breakdown", baseline_df=kdf)
    total = sdf["total_seconds"].replace(0, np.nan)
    pct = sdf[["graph_seconds", "eigensolver_seconds", "post_kmeans_seconds"]].div(total, axis=0) * 100
    pct.insert(0, "n", sdf["n"].values)
    return _save_results(pct, "runtime_bottleneck_percent.csv")


def run_all_experiments(random_state: int = 0) -> dict[str, pd.DataFrame]:
    outputs = {
        "synthetic": run_success_and_failure(random_state=random_state),
        "real": run_real_data(random_state=random_state),
        "noise": run_noise_experiment(random_state=random_state),
        "scaling": run_scaling_benchmark(random_state=random_state),
        "parameter_sensitivity": run_parameter_sensitivity(random_state=random_state),
        "eigengap": run_eigengap_study(random_state=random_state),
        "graph_ablation": run_graph_construction_ablation(random_state=random_state),
        "failure_taxonomy": run_failure_taxonomy(random_state=random_state),
        "sigma_sweep": run_sigma_sweep(random_state=random_state),
        "adaptive_sigma": run_adaptive_sigma(random_state=random_state),
        "sparsification": run_sparsification_tradeoff(random_state=random_state),
        "bottleneck": run_bottleneck_breakdown(random_state=random_state),
    }
    run_summary_figures()
    run_failure_comparison_figures(random_state=random_state)
    combined = pd.concat([outputs["synthetic"], outputs["real"]], ignore_index=True)
    _save_results(combined, "all_metrics.csv")
    return outputs
