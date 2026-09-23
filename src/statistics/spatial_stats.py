"""
Spatial statistics utilities for point patterns and minutiae configurations.
"""
import numpy as np
from scipy.spatial.distance import cdist, pdist

def nearest_neighbor_distances(points: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Computes k-th nearest neighbor distance for each point.
    If k=1, returns 1-NN distances.
    points: [N, 2] array.
    """
    if len(points) <= k:
        return np.array([], dtype=float)
    dmat = cdist(points, points)
    np.fill_diagonal(dmat, np.inf)
    sorted_dists = np.sort(dmat, axis=1)
    return sorted_dists[:, k - 1]

def minimum_separation(points: np.ndarray) -> float:
    """
    Returns the minimum pairwise distance between any two points in the set.
    """
    if len(points) <= 1:
        return np.inf
    nn = nearest_neighbor_distances(points, k=1)
    return float(np.min(nn))

def pairwise_distances(points: np.ndarray, max_pairs: int = 5000) -> np.ndarray:
    """
    Computes pairwise Euclidean distances between points. Subsamples if large.
    """
    if len(points) <= 1:
        return np.array([], dtype=float)
    dists = pdist(points)
    if len(dists) > max_pairs:
        idx = np.random.choice(len(dists), max_pairs, replace=False)
        return dists[idx]
    return dists

def local_densities(points: np.ndarray, radius: float = 30.0) -> np.ndarray:
    """
    Computes the number of neighboring minutiae within radius r for each point.
    """
    if len(points) <= 1:
        return np.zeros(len(points), dtype=float)
    dmat = cdist(points, points)
    np.fill_diagonal(dmat, np.inf)
    counts = np.sum(dmat <= radius, axis=1)
    return counts.astype(float)

def boundary_distances(points: np.ndarray, width: float = 416.0, height: float = 560.0) -> np.ndarray:
    """
    Computes the minimum distance from each point to the rectangular image boundary.
    """
    if len(points) == 0:
        return np.array([], dtype=float)
    dist_left = points[:, 0]
    dist_right = width - points[:, 0]
    dist_top = points[:, 1]
    dist_bottom = height - points[:, 1]
    return np.min(np.column_stack([dist_left, dist_right, dist_top, dist_bottom]), axis=1)

def ripleys_k(points: np.ndarray, radii: np.ndarray, area: float) -> np.ndarray:
    """
    Estimates Ripley's K-function K(r) for spatial point patterns.
    """
    n = len(points)
    if n <= 1:
        return np.zeros_like(radii)
    dmat = cdist(points, points)
    np.fill_diagonal(dmat, np.inf)
    
    k_vals = np.zeros(len(radii), dtype=float)
    for idx, r in enumerate(radii):
        count = np.sum(dmat <= r)
        k_vals[idx] = (area / (n * (n - 1))) * count
    return k_vals

def ripleys_l(points: np.ndarray, radii: np.ndarray, area: float) -> np.ndarray:
    """
    Estimates Besag's transformation of Ripley's K function: L(r) = sqrt(K(r) / pi) - r.
    Under Complete Spatial Randomness (CSR), L(r) = 0.
    L(r) > 0 indicates clustering; L(r) < 0 indicates inhibition / regularity.
    """
    k_vals = ripleys_k(points, radii, area)
    return np.sqrt(k_vals / np.pi) - radii
