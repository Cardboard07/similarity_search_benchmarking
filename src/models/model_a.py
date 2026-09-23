"""
Model A — Synthetic Fingerprint Identity Generator.
Generates completely new pristine fingerprint identities matching the statistical structure
of the real pristine population without copying, perturbing, or rotating existing fingerprints.
"""
from dataclasses import dataclass
import numpy as np
from src.statistics.distributions import EmpiricalSampler
from src.statistics.spatial_stats import minimum_separation
from src.models.spatial_model import (
    BaseSpatialModel,
    InhomogeneousPoissonInhibitionModel,
    AdaptiveKDESpatialModel,
    GaussianMixtureSpatialModel
)
from src.models.orientation_model import OrientationFieldModel
from src.models.type_model import MinutiaTypeModel

@dataclass
class SyntheticIdentity:
    pts: np.ndarray        # [N, 2] float32
    angles: np.ndarray     # [N] float32
    types: np.ndarray      # [N] int32
    qualities: np.ndarray  # [N] int32
    num_minutiae: int
    seed: int
    coef_u: np.ndarray
    coef_v: np.ndarray

class ModelA:
    """
    Model A: Identity Generator.
    Produces pristine fingerprint identities with realistic N, spatial point distribution,
    smooth orientation flow fields, circular residuals, and minutia types.
    """
    def __init__(
        self,
        spatial_model: BaseSpatialModel | None = None,
        count_sampler: EmpiricalSampler | None = None,
        orientation_model: OrientationFieldModel | None = None,
        type_model: MinutiaTypeModel | None = None,
        min_separation: float = 8.5
    ):
        self.spatial_model = spatial_model or InhomogeneousPoissonInhibitionModel(min_separation=min_separation)
        self.count_sampler = count_sampler
        self.orientation_model = orientation_model or OrientationFieldModel()
        self.type_model = type_model or MinutiaTypeModel()
        self.min_separation = min_separation

    def fit(self, pristine_fingerprints: list[dict]) -> "ModelA":
        """
        Fits all constituent statistical models using real pristine fingerprints.
        """
        counts = [len(fp['pts']) for fp in pristine_fingerprints]
        self.count_sampler = EmpiricalSampler(
            np.array(counts),
            discrete=True,
            clip_range=(16, 100)
        )
        
        all_pts = [fp['pts'] for fp in pristine_fingerprints if len(fp['pts']) > 0]
        self.spatial_model.fit(all_pts)
        self.orientation_model.fit(pristine_fingerprints)
        self.type_model.fit(pristine_fingerprints)
        
        return self

    def generate_identity(self, seed: int) -> SyntheticIdentity:
        """
        Generates a new pristine synthetic identity reproducibly using the given seed.
        """
        rng = np.random.default_rng(seed)
        
        # 1. Sample N minutiae
        n = int(self.count_sampler.sample(1, rng=rng)[0])
        
        # 2. Sample valid spatial locations with hard-core inhibition
        pts = self.spatial_model.sample(n, rng=rng)
        
        # Structural validation & repair if necessary
        if len(pts) < n:
            pts = self._repair_points(pts, n, rng)
            
        # 3. Sample unique smooth orientation field for this identity
        coef_u, coef_v = self.orientation_model.sample_identity_field_coefficients(rng=rng)
        
        # 4. Assign orientations via field + circular residuals
        angles = self.orientation_model.assign_orientations(pts, coef_u, coef_v, rng=rng)
        
        # 5. Assign minutia types
        types = self.type_model.assign_types(pts, rng=rng)
        
        # In pristine quality (q=3), minutia-level quality is 0 matching dataset
        qualities = np.zeros(len(pts), dtype=np.int32)
        
        return SyntheticIdentity(
            pts=pts.astype(np.float32),
            angles=angles.astype(np.float32),
            types=types.astype(np.int32),
            qualities=qualities,
            num_minutiae=len(pts),
            seed=seed,
            coef_u=coef_u,
            coef_v=coef_v
        )

    def _repair_points(self, pts: np.ndarray, target_n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Resamples missing points to guarantee exact target count while strictly adhering to boundaries.
        """
        pts_list = list(pts) if len(pts) > 0 else []
        attempts = 0
        while len(pts_list) < target_n and attempts < 100:
            candidate = np.array([
                rng.uniform(15.0, 400.0),
                rng.uniform(15.0, 480.0)
            ])
            if len(pts_list) == 0:
                pts_list.append(candidate)
            else:
                dists = np.linalg.norm(np.array(pts_list) - candidate, axis=1)
                if np.all(dists >= self.min_separation):
                    pts_list.append(candidate)
            attempts += 1
            
        return np.array(pts_list, dtype=np.float32)
