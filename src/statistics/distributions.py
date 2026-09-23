"""
Distribution models and sampling utilities (empirical and parametric).
"""
import numpy as np
from scipy import stats

class EmpiricalSampler:
    """
    1D Empirical distribution sampler using empirical CDF and inverse transform sampling.
    Supports linear interpolation / optional jitter for continuous quantities or exact discrete sampling.
    """
    def __init__(self, data: np.ndarray, discrete: bool = False, clip_range: tuple[float, float] | None = None):
        self.data = np.asarray(data, dtype=float)
        self.discrete = discrete
        self.clip_range = clip_range
        
        self.sorted_data = np.sort(self.data)
        self.n = len(self.sorted_data)
        self.probs = np.linspace(0.0, 1.0, self.n)
        
        # Summary statistics
        self.mean = float(np.mean(self.data))
        self.std = float(np.std(self.data))
        self.median = float(np.median(self.data))
        self.min_val = float(np.min(self.data))
        self.max_val = float(np.max(self.data))

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=size)
        if self.discrete:
            # Nearest or lower-bound index
            idx = np.searchsorted(self.probs, u, side='right')
            idx = np.clip(idx, 0, self.n - 1)
            samples = self.sorted_data[idx]
        else:
            samples = np.interp(u, self.probs, self.sorted_data)
            
        if self.clip_range is not None:
            samples = np.clip(samples, self.clip_range[0], self.clip_range[1])
            
        return samples

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.sorted_data, x, side='right') / float(self.n)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        return np.quantile(self.data, q)


class TruncatedNormalSampler:
    """
    Parametric sampler for truncated Gaussian variables.
    """
    def __init__(self, mean: float, std: float, lower: float, upper: float):
        self.mean = mean
        self.std = max(std, 1e-6)
        self.lower = lower
        self.upper = upper
        self.a = (lower - mean) / self.std
        self.b = (upper - mean) / self.std

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        # SciPy truncnorm rvs with numpy rng state
        seed = int(rng.integers(0, 2**31 - 1))
        return stats.truncnorm.rvs(self.a, self.b, loc=self.mean, scale=self.std, size=size, random_state=seed)
