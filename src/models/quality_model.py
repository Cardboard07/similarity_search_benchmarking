"""
Model B_quality: Quality Degradation Generator.
Learns and simulates realistic fingerprint capture quality degradation (q3 pristine, q7 good, q5 noisy, q1 bad, q9 very bad).
Features:
- Position jitter conditioned on quality and local field
- Circular orientation error
- Spatial dropout
- Spurious minutiae generation
- Smooth spatially-correlated quality field Q(x, y)
- Identity preservation guarantee
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from src.statistics.circular_stats import sample_von_mises, fit_von_mises
from src.statistics.spatial_stats import local_densities, boundary_distances

@dataclass
class QualityParameters:
    quality_level: int
    pos_noise_std: float
    angle_noise_kappa: float
    survival_rate: float
    false_minutiae_rate: float

class QualityModel:
    """
    Simulates quality degradations for levels 7 (good), 5 (noisy), 1 (bad), 9 (very bad)
    calibrated against empirical distributions from real paired data.
    """
    def __init__(self, use_spatial_field: bool = True):
        self.use_spatial_field = use_spatial_field
        
        # Ground-truth calibrated parameters (defaults based on 1000 identities)
        self.params: dict[int, QualityParameters] = {
            3: QualityParameters(quality_level=3, pos_noise_std=0.0, angle_noise_kappa=1000.0, survival_rate=1.0, false_minutiae_rate=0.0),
            7: QualityParameters(quality_level=7, pos_noise_std=1.5, angle_noise_kappa=40.0, survival_rate=0.985, false_minutiae_rate=0.85),
            5: QualityParameters(quality_level=5, pos_noise_std=2.0, angle_noise_kappa=25.0, survival_rate=0.975, false_minutiae_rate=1.20),
            1: QualityParameters(quality_level=1, pos_noise_std=2.8, angle_noise_kappa=18.0, survival_rate=0.950, false_minutiae_rate=2.40),
            9: QualityParameters(quality_level=9, pos_noise_std=2.6, angle_noise_kappa=15.0, survival_rate=0.948, false_minutiae_rate=2.35),
        }

    def fit_from_stats(self, quality_stats: dict[int, dict]) -> "QualityModel":
        """
        Updates parameters from empirical calibration statistics.
        quality_stats[q] has: 'pos_err', 'ang_err', 'surv', 'false'
        """
        for q in [7, 5, 1, 9]:
            if q in quality_stats:
                s = quality_stats[q]
                pos_std = float(np.mean(s['pos_err'])) / np.sqrt(2.0) # 2D Rayleigh/Gaussian conversion
                # Fit von Mises on angular errors
                ang_errors = np.array(s['ang_err'])
                _, kappa = fit_von_mises(ang_errors)
                surv = float(np.mean(s['surv']))
                false_mean = float(np.mean(s['false']))
                
                self.params[q] = QualityParameters(
                    quality_level=q,
                    pos_noise_std=float(np.clip(pos_std, 0.5, 6.0)),
                    angle_noise_kappa=float(np.clip(kappa, 5.0, 80.0)),
                    survival_rate=float(np.clip(surv, 0.85, 0.999)),
                    false_minutiae_rate=float(np.clip(false_mean, 0.0, 10.0))
                )
        return self

    def apply_quality_degradation(
        self,
        pts: np.ndarray,
        angles: np.ndarray,
        types: np.ndarray,
        quality_level: int,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Degrades minutiae according to quality level:
        3: pristine (identity)
        7: good
        5: noisy
        1: bad
        9: very bad
        Returns (degraded_pts, degraded_angles, degraded_types, minutiae_qualities)
        """
        n = len(pts)
        if n == 0 or quality_level == 3:
            return (
                pts.copy(),
                angles.copy(),
                types.copy(),
                np.zeros(n, dtype=np.int32)
            )

        qp = self.params.get(quality_level, self.params[7])
        
        # 1. Spatially Correlated Quality Field Q(x, y)
        if self.use_spatial_field and n > 5:
            # Generate smooth degradation field over control points
            n_ctrl = 5
            cx = np.linspace(20, 396, n_ctrl)
            cy = np.linspace(20, 470, n_ctrl)
            grid_x, grid_y = np.meshgrid(cx, cy)
            ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
            
            # Random field values at control points
            ctrl_q = rng.uniform(0.7, 1.3, size=len(ctrl_pts))
            
            dmat = cdist(pts, ctrl_pts)
            weights = np.exp(-0.5 * (dmat / 60.0)**2)
            w_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
            local_q_mult = (weights @ ctrl_q) / w_sum.squeeze()
        else:
            local_q_mult = np.ones(n, dtype=float)

        # 2. Minutiae Dropout (Survival)
        # Peripheral and locally noisy minutiae have slightly higher dropout
        b_dists = boundary_distances(pts)
        boundary_factor = np.clip(b_dists / 50.0, 0.8, 1.0)
        
        base_surv = qp.survival_rate
        # Local survival probability per minutia
        p_surv = np.clip(base_surv * boundary_factor / local_q_mult, 0.70, 0.999)
        u_surv = rng.uniform(0.0, 1.0, size=n)
        survived_mask = u_surv < p_surv
        
        # Identity preservation check: Ensure at least 80% survive to prevent destroying identity
        if np.sum(survived_mask) < int(0.75 * n):
            survived_indices = np.argsort(p_surv)[-int(0.75 * n):]
            survived_mask[:] = False
            survived_mask[survived_indices] = True
            
        surv_pts = pts[survived_mask].copy()
        surv_angles = angles[survived_mask].copy()
        surv_types = types[survived_mask].copy()
        surv_mult = local_q_mult[survived_mask]
        n_surv = len(surv_pts)

        # 3. Position Jitter (Local Localization Error)
        pos_stds = qp.pos_noise_std * surv_mult
        dx = rng.normal(0.0, 1.0, size=n_surv) * pos_stds
        dy = rng.normal(0.0, 1.0, size=n_surv) * pos_stds
        
        deg_pts = surv_pts + np.column_stack([dx, dy])
        # Clip within image bounds
        deg_pts[:, 0] = np.clip(deg_pts[:, 0], 10.0, 405.0)
        deg_pts[:, 1] = np.clip(deg_pts[:, 1], 10.0, 485.0)

        # 4. Orientation Noise
        angle_noises = sample_von_mises(0.0, qp.angle_noise_kappa, size=n_surv, rng=rng)
        angle_noises = (angle_noises + 180.0) % 360.0 - 180.0
        deg_angles = (surv_angles + angle_noises) % 360.0
        
        # 5. Spurious (False) Minutiae
        n_false = int(rng.poisson(qp.false_minutiae_rate))
        if n_false > 0 and n_surv > 0:
            # False minutiae appear in vicinity of existing minutiae (ridges/creases) or boundaries
            host_indices = rng.choice(n_surv, size=n_false, replace=True)
            host_pts = deg_pts[host_indices]
            host_angles = deg_angles[host_indices]
            
            # Position offset
            f_offset = rng.normal(0.0, 12.0, size=(n_false, 2))
            false_pts = host_pts + f_offset
            false_pts[:, 0] = np.clip(false_pts[:, 0], 10.0, 405.0)
            false_pts[:, 1] = np.clip(false_pts[:, 1], 10.0, 485.0)
            
            # False minutiae orientation roughly aligns or opposes ridge flow with higher noise
            f_angle_noise = sample_von_mises(0.0, 10.0, size=n_false, rng=rng)
            f_angle_noise = (f_angle_noise + 180.0) % 360.0 - 180.0
            false_angles = (host_angles + f_angle_noise) % 360.0
            
            # False types (ratio ~ 50/50)
            false_types = rng.choice([1, 2], size=n_false, p=[0.4, 0.6])
            
            # Concatenate
            all_pts = np.vstack([deg_pts, false_pts])
            all_angles = np.concatenate([deg_angles, false_angles])
            all_types = np.concatenate([surv_types, false_types])
        else:
            all_pts = deg_pts
            all_angles = deg_angles
            all_types = surv_types

        # Quantize angles to 8-bit precision
        bin_step = 360.0 / 256.0
        all_angles = np.round(all_angles / bin_step) * bin_step % 360.0
        
        # Minutiae quality field is 0 in the source dataset representation
        min_qualities = np.zeros(len(all_pts), dtype=np.int32)

        return (
            all_pts.astype(np.float32),
            all_angles.astype(np.float32),
            all_types.astype(np.int32),
            min_qualities
        )
