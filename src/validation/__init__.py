"""
Validation package for synthetic fingerprint minutiae:
- Statistical fidelity metrics (KS, Wasserstein, Circular, JS divergence)
- Identity preservation evaluation
- Novelty and anti-memorization testing
- Streaming large-scale validation engine
- Diagnostic publication plots
"""
from src.validation.statistical import StatisticalValidator, MetricComparison
from src.validation.identity import IdentityPreservationValidator
from src.validation.novelty import NoveltyValidator, NoveltyResult
from src.validation.scale_validation import StreamingScaleValidator, RunningMoments, ReservoirSampler
from src.validation.plots import (
    plot_count_distribution,
    plot_spatial_nn_distribution,
    plot_pairwise_distribution,
    plot_circular_polar_distribution,
    plot_synthetic_fingerprint_sample,
    plot_quality_degradation_curves
)
