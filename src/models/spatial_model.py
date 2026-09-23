"""
Spatial point process models for fingerprint minutiae.
Implements Adaptive KDE, Gaussian Mixture Model, and Inhomogeneous Poisson Process with Hard-Core Inhibition.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

class BaseSpatialModel(ABC):
    @abstractmethod
    def fit(self, all_points: list[np.ndarray]) -> "BaseSpatialModel":
        pass

    @abstractmethod
    def sample(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_log_likelihood(self, points: np.ndarray) -> float:
        pass


class GaussianMixtureSpatialModel(BaseSpatialModel):
    """
    Gaussian Mixture Model for 2D spatial minutiae distribution.
    """
    def __init__(self, n_components: int = 8, min_separation: float = 8.5):
        self.n_components = n_components
        self.min_separation = min_separation
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

    def fit(self, all_points: list[np.ndarray]) -> "GaussianMixtureSpatialModel":
        stacked = np.vstack(all_points)
        self.gmm.fit(stacked)
        return self

    def sample(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        accepted = []
        max_attempts = 50
        attempts = 0
        while len(accepted) < n_points and attempts < max_attempts:
            needed = (n_points - len(accepted)) * 3
            seed = int(rng.integers(0, 2**31 - 1))
            raw_pts, _ = self.gmm.sample(needed)
            for pt in raw_pts:
                # Boundary check
                if 10 <= pt[0] <= 405 and 10 <= pt[1] <= 485:
                    if len(accepted) == 0:
                        accepted.append(pt)
                    else:
                        dists = np.linalg.norm(np.array(accepted) - pt, axis=1)
                        if np.all(dists >= self.min_separation):
                            accepted.append(pt)
                    if len(accepted) == n_points:
                        break
            attempts += 1
            
        return np.array(accepted[:n_points], dtype=float)

    def evaluate_log_likelihood(self, points: np.ndarray) -> float:
        return float(np.mean(self.gmm.score_samples(points)))


class AdaptiveKDESpatialModel(BaseSpatialModel):
    """
    2D Kernel Density Estimation with boundary reflection/clipping and hard-core inhibition.
    """
    def __init__(self, bandwidth: float = 25.0, min_separation: float = 8.5):
        self.bandwidth = bandwidth
        self.min_separation = min_separation
        self.kde = None
        self.stacked_points = None

    def fit(self, all_points: list[np.ndarray]) -> "AdaptiveKDESpatialModel":
        self.stacked_points = np.vstack(all_points)
        # scipy gaussian_kde expects shape [dim, N]
        self.kde = stats.gaussian_kde(self.stacked_points.T, bw_method='scott')
        return self

    def sample(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        accepted = []
        attempts = 0
        max_attempts = 50
        while len(accepted) < n_points and attempts < max_attempts:
            needed = (n_points - len(accepted)) * 4
            seed = int(rng.integers(0, 2**31 - 1))
            raw_pts = self.kde.resample(needed, seed=seed).T
            for pt in raw_pts:
                if 10 <= pt[0] <= 405 and 10 <= pt[1] <= 485:
                    if len(accepted) == 0:
                        accepted.append(pt)
                    else:
                        dists = np.linalg.norm(np.array(accepted) - pt, axis=1)
                        if np.all(dists >= self.min_separation):
                            accepted.append(pt)
                    if len(accepted) == n_points:
                        break
            attempts += 1
            
        return np.array(accepted[:n_points], dtype=float)

    def evaluate_log_likelihood(self, points: np.ndarray) -> float:
        vals = self.kde.evaluate(points.T)
        return float(np.mean(np.log(np.clip(vals, 1e-12, None))))


class InhomogeneousPoissonInhibitionModel(BaseSpatialModel):
    """
    Inhomogeneous Poisson Point Process with continuous 2D intensity grid
    and Matérn-style hard-core inhibition (preventing minutiae closer than min_separation).
    """
    def __init__(
        self,
        grid_res: tuple[int, int] = (104, 140),
        min_separation: float = 8.5,
        smoothing_sigma: float = 12.0
    ):
        self.grid_res = grid_res # nx, ny
        self.min_separation = min_separation
        self.smoothing_sigma = smoothing_sigma
        
        self.x_edges = np.linspace(10, 405, grid_res[0] + 1)
        self.y_edges = np.linspace(10, 485, grid_res[1] + 1)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        
        self.intensity_grid = None
        self.flat_prob = None
        self.stacked_points = None

    def fit(self, all_points: list[np.ndarray]) -> "InhomogeneousPoissonInhibitionModel":
        self.stacked_points = np.vstack(all_points)
        # Compute 2D histogram
        H, _, _ = np.histogram2d(
            self.stacked_points[:, 0],
            self.stacked_points[:, 1],
            bins=[self.x_edges, self.y_edges]
        )
        
        # Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=self.smoothing_sigma / 4.0)
        # Add small baseline to prevent zero density inside the plausible area
        H_smooth = np.clip(H_smooth, 1e-5, None)
        
        self.intensity_grid = H_smooth
        self.flat_prob = (H_smooth / np.sum(H_smooth)).flatten()
        return self

    def sample(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        accepted = []
        nx, ny = self.grid_res
        max_attempts = 100
        dx = self.x_edges[1] - self.x_edges[0]
        dy = self.y_edges[1] - self.y_edges[0]
        
        attempts = 0
        while len(accepted) < n_points and attempts < max_attempts:
            needed = (n_points - len(accepted)) * 4
            cell_indices = rng.choice(nx * ny, size=needed, p=self.flat_prob)
            ix = cell_indices // ny
            iy = cell_indices % ny
            
            # Continuous uniform jitter within the chosen grid cell
            x_pts = self.x_edges[ix] + rng.uniform(0.0, dx, size=needed)
            y_pts = self.y_edges[iy] + rng.uniform(0.0, dy, size=needed)
            
            for pt in zip(x_pts, y_pts):
                if 10 <= pt[0] <= 405 and 10 <= pt[1] <= 485:
                    if len(accepted) == 0:
                        accepted.append(pt)
                    else:
                        dists = np.linalg.norm(np.array(accepted) - pt, axis=1)
                        if np.all(dists >= self.min_separation):
                            accepted.append(pt)
                    if len(accepted) == n_points:
                        break
            attempts += 1
            
        return np.array(accepted[:n_points], dtype=float)

    def evaluate_log_likelihood(self, points: np.ndarray) -> float:
        # Interpolate intensity at points
        ix = np.clip(np.searchsorted(self.x_edges, points[:, 0]) - 1, 0, self.grid_res[0] - 1)
        iy = np.clip(np.searchsorted(self.y_edges, points[:, 1]) - 1, 0, self.grid_res[1] - 1)
        probs = self.flat_prob[ix * self.grid_res[1] + iy]
        return float(np.mean(np.log(np.clip(probs, 1e-12, None))))
