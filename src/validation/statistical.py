"""
Statistical validation framework for synthetic fingerprint minutiae.
Computes quantitative distance metrics (Kolmogorov-Smirnov, Wasserstein, Jensen-Shannon, Circular distances)
comparing real and synthetic distributions across Model A and Model B phenomena.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from src.statistics.circular_stats import circular_abs_distance_deg

@dataclass
class MetricComparison:
    name: str
    real_mean: float
    synthetic_mean: float
    real_std: float
    synthetic_std: float
    ks_statistic: float | None
    ks_p_value: float | None
    wasserstein_distance: float
    histogram_intersection: float

class StatisticalValidator:
    """
    Computes statistical similarity metrics between real and synthetic fingerprint populations.
    """
    @staticmethod
    def compare_1d_continuous(
        real_data: np.ndarray,
        synth_data: np.ndarray,
        name: str,
        bins: int = 50
    ) -> MetricComparison:
        real_arr = np.asarray(real_data, dtype=float)
        synth_arr = np.asarray(synth_data, dtype=float)
        
        # KS test
        ks_res = stats.ks_2samp(real_arr, synth_arr)
        # Wasserstein 1D distance
        w_dist = float(stats.wasserstein_distance(real_arr, synth_arr))
        
        # Histogram intersection
        min_val = min(float(np.min(real_arr)), float(np.min(synth_arr)))
        max_val = max(float(np.max(real_arr)), float(np.max(synth_arr))) + 1e-6
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        h_real, _ = np.histogram(real_arr, bins=bin_edges, density=True)
        h_synth, _ = np.histogram(synth_arr, bins=bin_edges, density=True)
        dx = bin_edges[1] - bin_edges[0]
        overlap = float(np.sum(np.minimum(h_real, h_synth)) * dx)
        
        return MetricComparison(
            name=name,
            real_mean=float(np.mean(real_arr)),
            synthetic_mean=float(np.mean(synth_arr)),
            real_std=float(np.std(real_arr)),
            synthetic_std=float(np.std(synth_arr)),
            ks_statistic=float(ks_res.statistic),
            ks_p_value=float(ks_res.pvalue),
            wasserstein_distance=w_dist,
            histogram_intersection=float(np.clip(overlap, 0.0, 1.0))
        )

    @staticmethod
    def compare_discrete_proportions(
        real_labels: np.ndarray,
        synth_labels: np.ndarray,
        name: str
    ) -> dict:
        """
        Compares categorical distributions (e.g. Type 1 vs Type 2).
        """
        r_p1 = float(np.mean(np.asarray(real_labels) == 1))
        s_p1 = float(np.mean(np.asarray(synth_labels) == 1))
        
        # Chi-squared contingency test
        n_r = len(real_labels)
        n_s = len(synth_labels)
        table = [
            [int(r_p1 * n_r), int((1 - r_p1) * n_r)],
            [int(s_p1 * n_s), int((1 - s_p1) * n_s)]
        ]
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        
        # Jensen-Shannon divergence
        p = np.array([r_p1, 1.0 - r_p1])
        q = np.array([s_p1, 1.0 - s_p1])
        js_div = float(jensenshannon(p, q))
        
        return {
            'name': name,
            'real_type1_ratio': r_p1,
            'synthetic_type1_ratio': s_p1,
            'chi2_statistic': float(chi2),
            'p_value': float(p_val),
            'jensen_shannon_divergence': js_div
        }

    @staticmethod
    def compare_circular_distributions(
        real_angles_deg: np.ndarray,
        synth_angles_deg: np.ndarray,
        name: str
    ) -> dict:
        """
        Compares circular angular distributions.
        """
        from src.statistics.circular_stats import circular_mean_deg, circular_std_deg
        
        r_mean = circular_mean_deg(real_angles_deg)
        s_mean = circular_mean_deg(synth_angles_deg)
        r_std = circular_std_deg(real_angles_deg)
        s_std = circular_std_deg(synth_angles_deg)
        
        # Histogram on 36 bins [0, 360)
        bins = np.linspace(0, 360, 37)
        h_r, _ = np.histogram(real_angles_deg % 360.0, bins=bins, density=True)
        h_s, _ = np.histogram(synth_angles_deg % 360.0, bins=bins, density=True)
        overlap = float(np.sum(np.minimum(h_r, h_s)) * (360.0 / 36.0))
        
        js_div = float(jensenshannon(h_r + 1e-8, h_s + 1e-8))
        
        return {
            'name': name,
            'real_circular_mean_deg': r_mean,
            'synthetic_circular_mean_deg': s_mean,
            'real_circular_std_deg': r_std,
            'synthetic_circular_std_deg': s_std,
            'circular_histogram_overlap': float(np.clip(overlap, 0.0, 1.0)),
            'jensen_shannon_divergence': js_div
        }
