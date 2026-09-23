"""
Model B_angle: Impression / Orientation Generator.
Generates realistic alternate impressions / orientations for a given identity.
Models rigid transformation (scale, rotation, translation) plus smooth spatially-coherent local elastic deformation.
"""
from dataclasses import dataclass
import numpy as np
from src.statistics.circular_stats import sample_von_mises, fit_von_mises
from src.models.correspondence import CorrespondenceResult

@dataclass
class TransformationParameters:
    tx: float
    ty: float
    rotation_deg: float
    scale: float
    deformation_energy: float

class AngleModel:
    """
    Learns and applies impression variations (orientations 1, 2, 3) to a pristine fingerprint identity.
    """
    def __init__(
        self,
        correlation_length: float = 90.0,
        deformation_magnitude: float = 1.2
    ):
        self.correlation_length = correlation_length
        self.deformation_magnitude = deformation_magnitude
        
        # Rigid parameters learned from real paired impressions
        self.tx_std = 0.8
        self.ty_std = 0.8
        self.rot_std_deg = 0.2
        self.scale_std = 0.001
        
        # Angular residual noise
        self.residual_kappa = 25.0

    def fit(self, correspondence_results: list[CorrespondenceResult]) -> "AngleModel":
        """
        Fits rigid and elastic transformation distributions from correspondence results on real data.
        """
        tx_vals = [r.estimated_tx for r in correspondence_results]
        ty_vals = [r.estimated_ty for r in correspondence_results]
        rot_vals = [r.estimated_rotation_deg for r in correspondence_results]
        scale_vals = [r.estimated_scale for r in correspondence_results]
        ang_diffs = []
        for r in correspondence_results:
            if len(r.angular_differences) > 0:
                ang_diffs.extend(r.angular_differences)

        if len(tx_vals) > 0:
            self.tx_std = float(max(np.std(tx_vals), 0.1))
            self.ty_std = float(max(np.std(ty_vals), 0.1))
            self.rot_std_deg = float(max(np.std(rot_vals), 0.05))
            self.scale_std = float(max(np.std(scale_vals), 0.0005))
            
        if len(ang_diffs) > 0:
            _, kappa = fit_von_mises(np.array(ang_diffs))
            self.residual_kappa = float(np.clip(kappa, 10.0, 60.0))

        return self

    def apply_impression(
        self,
        pts: np.ndarray,
        angles: np.ndarray,
        types: np.ndarray,
        orientation_idx: int,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TransformationParameters]:
        """
        Transforms pristine minutiae into impression orientation_idx (1, 2, or 3).
        Orientation 1 is reference impression (minimal change / identity).
        Orientations 2 and 3 apply learned rigid + elastic deformation.
        """
        n = len(pts)
        if n == 0 or orientation_idx == 1:
            return (
                pts.copy(),
                angles.copy(),
                types.copy(),
                TransformationParameters(0.0, 0.0, 0.0, 1.0, 0.0)
            )

        # 1. Sample global rigid parameters
        tx = float(rng.normal(0.0, self.tx_std))
        ty = float(rng.normal(0.0, self.ty_std))
        rot_deg = float(rng.normal(0.0, self.rot_std_deg))
        scale = float(np.clip(rng.normal(1.0, self.scale_std), 0.98, 1.02))

        # Center points for rotation
        center = np.mean(pts, axis=0)
        centered = pts - center
        rad = np.radians(rot_deg)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rigid_pts = scale * (centered @ R.T) + center + np.array([tx, ty])

        # 2. Smooth spatially-correlated elastic deformation D(x, y)
        # Using Gaussian Random Field / RBF basis deformation
        n_ctrl = 6
        ctrl_x = np.linspace(20, 396, n_ctrl)
        ctrl_y = np.linspace(20, 470, n_ctrl)
        grid_x, grid_y = np.meshgrid(ctrl_x, ctrl_y)
        ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        
        # Sample random control point displacements
        ctrl_disp = rng.normal(0.0, self.deformation_magnitude, size=ctrl_pts.shape)
        
        # Gaussian kernel weights
        # dist: [N, n_ctrl*n_ctrl]
        from scipy.spatial.distance import cdist
        dmat = cdist(pts, ctrl_pts)
        weights = np.exp(-0.5 * (dmat / self.correlation_length)**2)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
        smooth_disp = (weights @ ctrl_disp) / weights_sum
        
        deformed_pts = rigid_pts + smooth_disp
        
        # Clip to valid bounding box
        deformed_pts[:, 0] = np.clip(deformed_pts[:, 0], 10.0, 405.0)
        deformed_pts[:, 1] = np.clip(deformed_pts[:, 1], 10.0, 485.0)

        # 3. Transform angles consistently + add circular residual
        transformed_angles = (angles + rot_deg) % 360.0
        angle_noise = sample_von_mises(0.0, self.residual_kappa, size=n, rng=rng)
        angle_noise = (angle_noise + 180.0) % 360.0 - 180.0
        final_angles = (transformed_angles + angle_noise) % 360.0
        
        # Quantize to 8-bit precision matching dataset
        bin_step = 360.0 / 256.0
        final_angles = np.round(final_angles / bin_step) * bin_step % 360.0

        def_energy = float(np.mean(np.linalg.norm(smooth_disp, axis=1)))
        params = TransformationParameters(
            tx=tx,
            ty=ty,
            rotation_deg=rot_deg,
            scale=scale,
            deformation_energy=def_energy
        )

        return (
            deformed_pts.astype(np.float32),
            final_angles.astype(np.float32),
            types.copy(),
            params
        )
