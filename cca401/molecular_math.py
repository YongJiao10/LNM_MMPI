"""PLS, spin geometry and BH calculations shared by current molecular analyses."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
ENRICHR_LIBRARIES = ('GO_Biological_Process_2023','KEGG_2021_Human','Reactome_2022')

def zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.zeros_like(values)
    return (values - values.mean()) / std

def standardize_predictors(X: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(zscore_1d, 0, X)

def fit_single_component_pls_standardized(
    Xz: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray | float]:
    yz = zscore_1d(y).reshape(-1, 1)
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(Xz, yz)

    x_weights = np.asarray(pls.x_weights_[:, 0], dtype=float)
    gene_score_map = Xz @ x_weights
    gene_score_map = zscore_1d(gene_score_map)
    observed_r = float(pearsonr(gene_score_map, yz.ravel()).statistic)
    observed_cov = float(np.cov(pls.x_scores_[:, 0], pls.y_scores_[:, 0], ddof=1)[0, 1])

    return {
        "x_weights": x_weights,
        "gene_score_map": gene_score_map,
        "observed_r": observed_r,
        "observed_cov": observed_cov,
    }

def build_spin_model(n_perm: int, seed: int):
    from brainspace import datasets as bs_datasets
    from brainspace.null_models import SpinPermutations
    lh_labels, rh_labels = bs_datasets.load_parcellation("schaefer", scale=100, join=False)
    lh_sphere, rh_sphere = bs_datasets.load_conte69(as_sphere=True, join=False)

    def centroids(points: np.ndarray, labels: np.ndarray, unique_labels: Iterable[int]) -> np.ndarray:
        output = []
        for lab in unique_labels:
            verts = points[labels == lab]
            output.append(np.asarray(verts).mean(axis=0))
        return np.vstack(output)

    lh_points = np.asarray(lh_sphere.Points)
    rh_points = np.asarray(rh_sphere.Points)
    lh_centroids = centroids(lh_points, lh_labels, range(1, 51))
    rh_centroids = centroids(rh_points, rh_labels, range(51, 101))

    model = SpinPermutations(n_rep=n_perm, random_state=seed)
    model.fit(lh_centroids, rh_centroids)
    return model

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = ranked[idx] * n / rank
        prev = min(prev, value)
        adjusted[idx] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out
