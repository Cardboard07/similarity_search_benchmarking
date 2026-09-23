"""
Large-scale streaming validation engine.
Computes online moments (Welford's algorithm), streaming histograms, and reservoir sampling
over millions of synthetic observations without loading the dataset into RAM.
"""
import numpy as np

class RunningMoments:
    """
    Online mean and variance computation using Welford's algorithm.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = np.inf
        self.max_val = -np.inf

    def update(self, val: float):
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

    def update_batch(self, batch: np.ndarray):
        arr = np.asarray(batch, dtype=float)
        for v in arr:
            self.update(v)

    @property
    def variance(self) -> float:
        return float(self.M2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


class ReservoirSampler:
    """
    Reservoir sampling algorithm (Algorithm R) to maintain a uniformly distributed
    fixed-size sample of size K over an unbounded stream of observations.
    """
    def __init__(self, capacity: int = 10000, rng: np.random.Generator | None = None):
        self.capacity = capacity
        self.rng = rng or np.random.default_rng(42)
        self.sample = []
        self.total_seen = 0

    def feed(self, item):
        self.total_seen += 1
        if len(self.sample) < self.capacity:
            self.sample.append(item)
        else:
            j = self.rng.integers(0, self.total_seen)
            if j < self.capacity:
                self.sample[j] = item

    def get_sample_array(self) -> np.ndarray:
        return np.array(self.sample)


class StreamingScaleValidator:
    """
    Aggregates statistical moments and histograms across large streaming batches.
    """
    def __init__(self, reservoir_size: int = 20000):
        self.count_moments = RunningMoments()
        self.count_reservoir = ReservoirSampler(capacity=reservoir_size)
        self.spatial_reservoir = ReservoirSampler(capacity=reservoir_size)
        self.orientation_reservoir = ReservoirSampler(capacity=reservoir_size)
        self.type_counts = {1: 0, 2: 0}
        self.total_observations = 0

    def process_batch(
        self,
        minutiae_batch: np.ndarray,      # [N, MAX_M, 5]
        num_minutiae_batch: np.ndarray,  # [N]
        qualities_batch: np.ndarray      # [N]
    ):
        """
        Updates streaming statistics from an in-memory batch.
        """
        n_obs = len(num_minutiae_batch)
        self.total_observations += n_obs
        self.count_moments.update_batch(num_minutiae_batch)

        for i in range(n_obs):
            c = int(num_minutiae_batch[i])
            self.count_reservoir.feed(c)
            
            # Subsample some minutiae for spatial/orientation reservoirs
            if c > 0:
                raw_m = minutiae_batch[i, :c]
                # sample up to 2 minutiae per fingerprint to avoid bias
                idx = np.random.choice(c, size=min(c, 2), replace=False)
                for j in idx:
                    self.spatial_reservoir.feed(raw_m[j, :2])
                    self.orientation_reservoir.feed(raw_m[j, 2])
                    m_type = int(raw_m[j, 3])
                    if m_type in self.type_counts:
                        self.type_counts[m_type] += 1

    def get_summary(self) -> dict:
        total_types = sum(self.type_counts.values())
        return {
            'total_observations_processed': self.total_observations,
            'minutiae_count': {
                'mean': float(self.count_moments.mean),
                'std': float(self.count_moments.std),
                'min': float(self.count_moments.min_val) if self.count_moments.count > 0 else 0,
                'max': float(self.count_moments.max_val) if self.count_moments.count > 0 else 0,
            },
            'type_proportions': {
                'type_1': float(self.type_counts[1] / total_types) if total_types > 0 else 0.0,
                'type_2': float(self.type_counts[2] / total_types) if total_types > 0 else 0.0
            }
        }
