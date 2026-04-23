import numpy as np
import networkx as nx
from sklearn import datasets
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def make_synthetic_datasets(random_state: int = 0):
    X1, y1 = datasets.make_moons(n_samples=400, noise=0.08, random_state=random_state)
    X2, y2 = datasets.make_circles(n_samples=400, factor=0.45, noise=0.06, random_state=random_state)
    X3, y3 = datasets.make_blobs(n_samples=450, centers=3, cluster_std=[1.0, 1.2, 1.0], random_state=random_state)
    X4, y4 = datasets.make_blobs(n_samples=450, centers=[[0,0],[0.8,0.8],[1.6,1.6]], cluster_std=[0.9,0.9,0.9], random_state=random_state)
    return {
        'two_moons': (X1, y1),
        'circles': (X2, y2),
        'blobs': (X3, y3),
        'overlapping_blobs_failure': (X4, y4),
    }


def load_iris_data():
    ds = load_iris()
    return StandardScaler().fit_transform(ds.data), ds.target


def load_digits_subset(classes=(0,1,3,6), n_per_class=80, random_state: int = 0):
    ds = load_digits()
    X = ds.data
    y = ds.target
    rs = np.random.default_rng(random_state)
    xs=[]; ys=[]
    for c in classes:
        idx = np.where(y == c)[0]
        rs.shuffle(idx)
        idx = idx[:n_per_class]
        xs.append(X[idx])
        ys.append(y[idx])
    X = np.vstack(xs)
    y = np.concatenate(ys)
    X = StandardScaler().fit_transform(X)
    return X, y


def digits_pca_2d(classes=(0,1,3,6), n_per_class=80, random_state: int = 0):
    X, y = load_digits_subset(classes=classes, n_per_class=n_per_class, random_state=random_state)
    X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
    return X2, y


def karate_graph():
    G = nx.karate_club_graph()
    y = np.array([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()])
    return G, y
