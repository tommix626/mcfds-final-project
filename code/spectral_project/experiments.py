from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .clustering import (
    dense_rbf_affinity,
    first_laplacian_eigenvalues,
    kmeans_baseline,
    knn_rbf_affinity_sparse,
    mutual_knn_rbf_affinity_sparse,
    ng_jordan_weiss,
    spectral_from_affinity,
    spectral_on_graph,
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
    heatmap,
    line_noise,
    line_scaling,
    multi_panel_scatter,
    scatter_clusters,
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
        ("failure", failure, FAILURE_NEIGHBORS),
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
            scatter_clusters(figs / f"{name}_spectral.png", X_plot, spec.labels, f"{name}: spectral clustering")
            scatter_clusters(figs / f"{name}_kmeans.png", X_plot, km.labels, f"{name}: k-means")

    df = pd.DataFrame(rows)
    return _save_results(df, "synthetic_metrics.csv")



def run_real_data(random_state: int = 0) -> pd.DataFrame:
    rows: list[dict] = []

    X_iris, y_iris = load_iris_data()
    spec_iris = ng_jordan_weiss(X_iris, n_clusters=3, n_neighbors=12, random_state=random_state)
    km_iris = kmeans_baseline(X_iris, n_clusters=3, random_state=random_state)
    rows.append(_record_rows("iris", "real", "Spectral should be competitive; gains may be modest on nearly convex classes.", "spectral", y_iris, spec_iris.labels, spec_iris.timings))
    rows.append(_record_rows("iris", "real", "Spectral should be competitive; gains may be modest on nearly convex classes.", "kmeans", y_iris, km_iris.labels, km_iris.timings))

    # Digit subset visualizations requested in the TODO/workup.
    X2, y2 = digits_pca_2d(classes=(0, 1, 3, 6), random_state=random_state)
    spec2 = ng_jordan_weiss(X2, n_clusters=4, n_neighbors=10, random_state=random_state)
    km2 = kmeans_baseline(X2, n_clusters=4, random_state=random_state)
    scatter_clusters(figures_dir() / "digits_2d_spectral.png", X2, spec2.labels, "Digits {0,1,3,6} PCA(2): spectral")
    scatter_clusters(figures_dir() / "digits_2d_kmeans.png", X2, km2.labels, "Digits {0,1,3,6} PCA(2): k-means")

    X_all, y_all = load_all_digits()
    for name, classes, k, neighbors in DIGITS_SPECS:
        if name == "digits_all":
            Xd, yd = X_all, y_all
        else:
            Xd, yd = load_digits_subset(classes=classes, random_state=random_state)
        spec = ng_jordan_weiss(Xd, n_clusters=k, n_neighbors=neighbors, random_state=random_state)
        km = kmeans_baseline(Xd, n_clusters=k, random_state=random_state)
        expectation = "Digits form semantically meaningful but imperfect clusters; spectral should help more on harder subset geometry."
        rows.append(_record_rows(name, "digits", expectation, "spectral", yd, spec.labels, spec.timings))
        rows.append(_record_rows(name, "digits", expectation, "kmeans", yd, km.labels, km.timings))

    Gk, yk = karate_graph()
    Ak = nx.to_numpy_array(Gk, weight="weight")
    spec_g = spectral_on_graph(Ak, n_clusters=2, random_state=random_state)
    km_g = kmeans_baseline(Ak, n_clusters=2, random_state=random_state)
    rows.append(_record_rows("karate_graph", "graph", "Spectral should recover the post-split factions more naturally than adjacency-row k-means.", "spectral", yk, spec_g.labels, spec_g.timings))
    rows.append(_record_rows("karate_graph", "graph", "Spectral should recover the post-split factions more naturally than adjacency-row k-means.", "kmeans_on_adjacency", yk, km_g.labels, km_g.timings))

    # Graph figure
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(Gk, seed=random_state)
    plt.figure(figsize=(5.4, 4.4))
    nx.draw_networkx(Gk, pos=pos, node_color=spec_g.labels, with_labels=False, node_size=160)
    plt.title("Karate club: spectral clustering on graph")
    plt.tight_layout()
    plt.savefig(figures_dir() / "karate_spectral.png", dpi=220)
    plt.close()

    # SBM benchmark requested by PDF examples.
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

    # Success/failure ARI summary.
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

    # Save a compact json summary for the README / downstream use.
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
    """Phase-diagram style sweep over noise and kNN graph degree on two moons."""
    from sklearn import datasets
    noises = [0.02, 0.08, 0.14, 0.20, 0.28, 0.36]
    neighbors = [3, 5, 8, 12, 18, 28]
    rows: list[dict] = []
    mat = np.zeros((len(noises), len(neighbors)))
    for i, noise in enumerate(noises):
        X, y = datasets.make_moons(n_samples=500, noise=noise, random_state=random_state)
        km = kmeans_baseline(X, n_clusters=2, random_state=random_state)
        km_ari = summarize(y, km.labels)["ari"]
        for j, nn in enumerate(neighbors):
            try:
                spec = ng_jordan_weiss(X, n_clusters=2, n_neighbors=nn, random_state=random_state)
                ari = summarize(y, spec.labels)["ari"]
            except Exception:
                spec = ng_jordan_weiss(X, n_clusters=2, n_neighbors=nn, random_state=random_state, dense_threshold=X.shape[0] + 1)
                ari = summarize(y, spec.labels)["ari"]
            mat[i, j] = ari
            rows.append({"dataset": "two_moons", "noise": noise, "n_neighbors": nn, "spectral_ari": ari, "kmeans_ari_at_same_noise": km_ari})
    heatmap(figures_dir() / "parameter_sensitivity_heatmap.png", mat, [str(x) for x in neighbors], [str(x) for x in noises], "Spectral clustering sensitivity on two moons", "kNN graph degree", "Noise standard deviation", "ARI")
    return _save_results(pd.DataFrame(rows), "parameter_sensitivity_metrics.csv")


def run_eigengap_study(random_state: int = 0) -> pd.DataFrame:
    """Compare first Laplacian eigenvalues on easy and hard examples."""
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
    eigenvalue_plot(figures_dir() / "eigengap_study.png", eigs_dict, "Small Laplacian eigenvalues across datasets")
    return _save_results(pd.DataFrame(rows), "eigengap_metrics.csv")


def run_graph_construction_ablation(random_state: int = 0) -> pd.DataFrame:
    """Compare kNN, mutual-kNN, and dense RBF graph construction choices."""
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
        graphs = {"knn_rbf": knn_rbf_affinity_sparse(X, n_neighbors=nn), "mutual_knn_rbf": mutual_knn_rbf_affinity_sparse(X, n_neighbors=nn), "dense_rbf": dense_rbf_affinity(X)}
        for graph_name, W in graphs.items():
            try:
                spec = spectral_from_affinity(W, n_clusters=k, random_state=random_state)
                rows.append({"dataset": name, "graph": graph_name, **summarize(y, spec.labels)})
            except Exception as e:
                rows.append({"dataset": name, "graph": graph_name, "accuracy": np.nan, "ari": np.nan, "nmi": np.nan, "purity": np.nan, "error": str(e)})
    df = pd.DataFrame(rows)
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="dataset", columns="graph", values="ari")
    pivot.plot(kind="bar", figsize=(7.0, 4.4))
    plt.ylabel("ARI")
    plt.title("Graph construction ablation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir() / "graph_construction_ablation.png", dpi=220)
    plt.close()
    return _save_results(df, "graph_construction_ablation.csv")


def run_failure_taxonomy(random_state: int = 0) -> pd.DataFrame:
    """Curate failure archetypes: disconnected graph, bridge connection, and varying density."""
    from sklearn import datasets
    rows: list[dict] = []
    panels = []
    X1, y1 = datasets.make_moons(n_samples=200, noise=0.08, random_state=random_state)
    # Use a deliberately underspecified graph.  Keeping n small forces the dense eigensolver, avoiding ARPACK non-convergence on a fragmented graph.
    spec1 = ng_jordan_weiss(X1, n_clusters=2, n_neighbors=2, random_state=random_state, dense_threshold=250)
    rows.append({"failure_mode": "too_few_neighbors_fragmentation", "dataset": "moons", "n_neighbors": 2, **summarize(y1, spec1.labels)})
    panels.append((X1, spec1.labels, "Too few neighbors"))
    X2, y2, k2, _ = make_failure_datasets(random_state=random_state)["bridge_failure"]
    spec2 = ng_jordan_weiss(X2, n_clusters=k2, n_neighbors=14, random_state=random_state)
    rows.append({"failure_mode": "bridge_connection", "dataset": "bridge", "n_neighbors": 14, **summarize(y2, spec2.labels)})
    panels.append((X2, spec2.labels, "Bridge connection"))
    rs = np.random.default_rng(random_state)
    dense = rs.normal(loc=[-2.0, 0.0], scale=0.22, size=(90, 2))
    diffuse = rs.normal(loc=[1.7, 0.0], scale=0.85, size=(90, 2))
    tiny = rs.normal(loc=[0.0, 2.2], scale=0.18, size=(25, 2))
    X3 = np.vstack([dense, diffuse, tiny])
    y3 = np.concatenate([np.zeros(len(dense), dtype=int), np.ones(len(diffuse), dtype=int), np.full(len(tiny), 2, dtype=int)])
    spec3 = ng_jordan_weiss(X3, n_clusters=3, n_neighbors=10, random_state=random_state, dense_threshold=250)
    rows.append({"failure_mode": "varying_density_imbalance", "dataset": "density_imbalance", "n_neighbors": 10, **summarize(y3, spec3.labels)})
    panels.append((X3, spec3.labels, "Varying density"))
    multi_panel_scatter(figures_dir() / "failure_taxonomy.png", panels, "Three spectral clustering failure archetypes")
    return _save_results(pd.DataFrame(rows), "failure_taxonomy_metrics.csv")


def run_bottleneck_breakdown(random_state: int = 0) -> pd.DataFrame:
    """More explicit runtime bottleneck figure derived from the scaling benchmark."""
    df = pd.read_csv(results_dir() / "scaling_metrics.csv")
    sdf = df[df["method"] == "spectral"].copy()
    stacked_runtime(figures_dir() / "runtime_bottleneck_breakdown.png", sdf, "Spectral pipeline bottleneck breakdown")
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
        "bottleneck": run_bottleneck_breakdown(random_state=random_state),
    }
    run_summary_figures()
    combined = pd.concat([outputs["synthetic"], outputs["real"]], ignore_index=True)
    _save_results(combined, "all_metrics.csv")
    return outputs
