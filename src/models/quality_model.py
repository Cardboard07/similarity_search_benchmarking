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
        
        # Ground-truth calibrated parameters (defaults based on 1000 real identities)
        self.params: dict[int, QualityParameters] = {
            3: QualityParameters(quality_level=3, pos_noise_std=0.0, angle_noise_kappa=100000.0, survival_rate=1.0, false_minutiae_rate=0.0),
            7: QualityParameters(quality_level=7, pos_noise_std=0.80, angle_noise_kappa=686.0, survival_rate=0.990, false_minutiae_rate=0.51),
            5: QualityParameters(quality_level=5, pos_noise_std=0.78, angle_noise_kappa=533.0, survival_rate=0.988, false_minutiae_rate=0.60),
            1: QualityParameters(quality_level=1, pos_noise_std=0.65, angle_noise_kappa=994.0, survival_rate=0.982, false_minutiae_rate=0.82),
            9: QualityParameters(quality_level=9, pos_noise_std=0.45, angle_noise_kappa=1800.0, survival_rate=0.981, false_minutiae_rate=0.86),
        }

    def fit_from_stats(self, quality_stats: dict[int, dict]) -> "QualityModel":
        """
        Updates parameters from empirical calibration statistics.
        quality_stats[q] has: 'pos_err_mean', 'fitted_kappa', 'surv', 'false'
        """
        for q in [7, 5, 1, 9]:
            if q in quality_stats:
                s = quality_stats[q]
                pos_err = float(s.get('pos_err_mean', 1.5))
                pos_std = pos_err * 0.48 # RBF smooth field scale factor
                kappa = float(s.get('fitted_kappa', 800.0))
                surv = float(s.get('surv', 0.985))
                false_mean = float(s.get('false', 0.6))
                
                self.params[q] = QualityParameters(
                    quality_level=q,
                    pos_noise_std=float(np.clip(pos_std, 0.2, 3.0)),
                    angle_noise_kappa=float(np.clip(kappa, 100.0, 50000.0)),
                    survival_rate=float(np.clip(surv, 0.90, 0.999)),
                    false_minutiae_rate=float(np.clip(false_mean, 0.0, 5.0))
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
        
        # 1. Minutiae Dropout (Survival)
        b_dists = boundary_distances(pts)
        boundary_factor = np.clip(b_dists / 50.0, 0.95, 1.0)
        p_surv = np.clip(qp.survival_rate * boundary_factor, 0.92, 0.999)
        u_surv = rng.uniform(0.0, 1.0, size=n)
        survived_mask = u_surv < p_surv
        
        # Identity preservation check: Ensure at least 85% survive to prevent destroying identity
        if np.sum(survived_mask) < int(0.85 * n):
            survived_indices = np.argsort(p_surv)[-int(0.85 * n):]
            survived_mask[:] = False
            survived_mask[survived_indices] = True
            
        surv_pts = pts[survived_mask].copy()
        surv_angles = angles[survived_mask].copy()
        surv_types = types[survived_mask].copy()
        n_surv = len(surv_pts)

        # 2. Smooth Spatially-Correlated Elastic Displacement and Angular Fields
        if self.use_spatial_field and n_surv > 5:
            n_ctrl = 5
            cx = np.linspace(20, 396, n_ctrl)
            cy = np.linspace(20, 470, n_ctrl)
            grid_x, grid_y = np.meshgrid(cx, cy)
            ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
            
            dmat = cdist(surv_pts, ctrl_pts)
            weights = np.exp(-0.5 * (dmat / 80.0)**2)
            weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
            
            # Smooth displacement field (RBF control grid)
            ctrl_disp = rng.normal(0.0, qp.pos_noise_std * 2.1, size=ctrl_pts.shape)
            smooth_disp = (weights @ ctrl_disp) / weights_sum
            
            # Smooth angular variation field (matching empirical ~1.5 deg)
            ang_scale_rad = 1.0 / np.sqrt(qp.angle_noise_kappa) if qp.angle_noise_kappa > 0 else 0.05
            ang_scale_deg = np.degrees(ang_scale_rad)
            ctrl_ang = rng.normal(0.0, ang_scale_deg * 1.5, size=len(ctrl_pts))
            smooth_ang = (weights @ ctrl_ang) / weights_sum.squeeze()
        else:
            smooth_disp = np.zeros((n_surv, 2), dtype=float)
            smooth_ang = np.zeros(n_surv, dtype=float)

        # 3. Position Jitter (Smooth field + small independent local jitter)
        indep_jitter = rng.normal(0.0, 0.20, size=(n_surv, 2))
        deg_pts = surv_pts + smooth_disp + indep_jitter
        deg_pts[:, 0] = np.clip(deg_pts[:, 0], 10.0, 405.0)
        deg_pts[:, 1] = np.clip(deg_pts[:, 1], 10.0, 485.0)

        # 4. Orientation Noise (Smooth field + small independent residual)
        indep_ang = rng.normal(0.0, 0.20, size=n_surv)
        deg_angles = (surv_angles + smooth_ang + indep_ang) % 360.0
        
        # 5. Spurious (False) Minutiae
        n_false = int(rng.poisson(qp.false_minutiae_rate))
        if n_false > 0 and n_surv > 0:
            host_indices = rng.choice(n_surv, size=n_false, replace=True)
            f_offset = rng.normal(0.0, 15.0, size=(n_false, 2))
            false_pts = deg_pts[host_indices] + f_offset
            false_pts[:, 0] = np.clip(false_pts[:, 0], 10.0, 405.0)
            false_pts[:, 1] = np.clip(false_pts[:, 1], 10.0, 485.0)
            
            # False minutiae orientation roughly aligns with ridge flow
            f_angle_noise = sample_von_mises(0.0, 80.0, size=n_false, rng=rng)
            f_angle_noise = (f_angle_noise + 180.0) % 360.0 - 180.0
            false_angles = (deg_angles[host_indices] + f_angle_noise) % 360.0
            false_types = rng.choice([1, 2], size=n_false, p=[0.4, 0.6])
            
            all_pts = np.vstack([deg_pts, false_pts])
            all_angles = np.concatenate([deg_angles, false_angles])
            all_types = np.concatenate([surv_types, false_types])
        else:
            all_pts = deg_pts
            all_angles = deg_angles
            all_types = surv_types

        # Quantize angles to 8-bit precision matching real dataset
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
