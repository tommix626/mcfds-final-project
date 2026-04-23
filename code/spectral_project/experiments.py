from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from .data import make_synthetic_datasets, load_iris_data, digits_pca_2d, load_digits_subset, karate_graph
from .clustering import ng_jordan_weiss, spectral_on_graph
from .metrics import summarize
from .plots import scatter_clusters, line_noise
from .utils import figures_dir, results_dir


def run_synthetic(random_state: int = 0):
    figs = figures_dir()
    rows = []
    datasets = make_synthetic_datasets(random_state=random_state)
    cluster_counts = {'two_moons':2, 'circles':2, 'blobs':3, 'overlapping_blobs_failure':3}
    for name, (X, y) in datasets.items():
        spec = ng_jordan_weiss(X, n_clusters=cluster_counts[name], n_neighbors=12, random_state=random_state)
        km = KMeans(n_clusters=cluster_counts[name], n_init=20, random_state=random_state, algorithm='lloyd').fit_predict(X)
        ms = summarize(y, spec.labels)
        mk = summarize(y, km)
        rows.append({'dataset':name, 'method':'spectral', **ms})
        rows.append({'dataset':name, 'method':'kmeans', **mk})
        scatter_clusters(figs / f'{name}_spectral.png', X, spec.labels, f'{name}: spectral clustering')
        scatter_clusters(figs / f'{name}_kmeans.png', X, km, f'{name}: k-means')
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / 'synthetic_metrics.csv', index=False)
    return df


def run_iris(random_state: int = 0):
    X, y = load_iris_data()
    spec = ng_jordan_weiss(X, n_clusters=3, n_neighbors=12, random_state=random_state)
    km = KMeans(n_clusters=3, n_init=20, random_state=random_state, algorithm='lloyd').fit_predict(X)
    df = pd.DataFrame([
        {'dataset':'iris','method':'spectral', **summarize(y, spec.labels)},
        {'dataset':'iris','method':'kmeans', **summarize(y, km)},
    ])
    df.to_csv(results_dir() / 'iris_metrics.csv', index=False)
    return df


def run_digits(random_state: int = 0):
    # For reliability and speed in a lightweight handoff, run the displayed experiment on a PCA-reduced digits subset.
    X2, y = digits_pca_2d(n_per_class=60, random_state=random_state)
    spec2 = ng_jordan_weiss(X2, n_clusters=4, n_neighbors=10, random_state=random_state)
    km2 = KMeans(n_clusters=4, n_init=20, random_state=random_state, algorithm='lloyd').fit_predict(X2)
    df = pd.DataFrame([
        {'dataset':'digits_0136_pca2','method':'spectral', **summarize(y, spec2.labels)},
        {'dataset':'digits_0136_pca2','method':'kmeans', **summarize(y, km2)},
    ])
    df.to_csv(results_dir() / 'digits_metrics.csv', index=False)
    scatter_clusters(figures_dir() / 'digits_2d_spectral.png', X2, spec2.labels, 'Digits subset (PCA to 2D): spectral')
    scatter_clusters(figures_dir() / 'digits_2d_kmeans.png', X2, km2, 'Digits subset (PCA to 2D): k-means')
    return df


def run_karate(random_state: int = 0):
    G, y = karate_graph()
    A = nx.to_numpy_array(G)
    spec = spectral_on_graph(A, n_clusters=2, random_state=random_state)
    pos = nx.spring_layout(G, seed=random_state)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    nx.draw_networkx(G, pos=pos, node_color=spec.labels, with_labels=False, node_size=140)
    plt.title('Karate club: spectral clustering on graph')
    plt.tight_layout()
    plt.savefig(figures_dir() / 'karate_spectral.png', dpi=200)
    plt.close()
    df = pd.DataFrame([
        {'dataset':'karate','method':'spectral', **summarize(y, spec.labels)}
    ])
    df.to_csv(results_dir() / 'karate_metrics.csv', index=False)
    return df


def run_noise_experiment(random_state: int = 0):
    from sklearn import datasets
    X, y = datasets.make_moons(n_samples=500, noise=0.06, random_state=random_state)
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4]
    spec_scores = []
    km_scores = []
    for s in sigmas:
        rs = np.random.default_rng(random_state)
        Xn = X + rs.normal(scale=s, size=X.shape)
        spec = ng_jordan_weiss(Xn, n_clusters=2, n_neighbors=12, random_state=random_state)
        km = KMeans(n_clusters=2, n_init=20, random_state=random_state, algorithm='lloyd').fit_predict(Xn)
        spec_scores.append(summarize(y, spec.labels)['ari'])
        km_scores.append(summarize(y, km)['ari'])
    line_noise(figures_dir() / 'noise_moons_ari.png', sigmas, spec_scores, km_scores,
               'Two moons under additive Gaussian noise', 'ARI')
    return pd.DataFrame({'sigma': sigmas, 'spectral_ari': spec_scores, 'kmeans_ari': km_scores})


def run_all_experiments(random_state: int = 0):
    out = []
    out.append(run_synthetic(random_state))
    out.append(run_iris(random_state))
    out.append(run_digits(random_state))
    out.append(run_karate(random_state))
    noise = run_noise_experiment(random_state)
    noise.to_csv(results_dir() / 'noise_metrics.csv', index=False)
    combined = pd.concat(out, ignore_index=True)
    combined.to_csv(results_dir() / 'all_metrics.csv', index=False)
    print(combined)
    print(noise)
