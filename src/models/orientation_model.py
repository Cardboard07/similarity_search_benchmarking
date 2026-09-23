"""
Smooth 2D orientation field model with circular residual noise.
Uses bivariate polynomial / PCA expansion in doubled-angle representation (cos(2*theta), sin(2*theta))
to preserve global ridge flow smoothness, singular points, and local circular residual variation.
"""
import numpy as np
from src.statistics.circular_stats import (
    to_doubled_angle_vector,
    from_doubled_angle_vector,
    fit_von_mises,
    sample_von_mises,
    circular_distance_deg,
    ridge_flow_distance_deg
)

class OrientationFieldModel:
    """
    Learns 2D orientation fields and residual angular noise across fingerprint identities.
    Uses polynomial basis in doubled-angle space plus PCA basis for identity-level field diversity.
    """
    def __init__(self, poly_order: int = 4, n_pca_components: int = 4):
        self.poly_order = poly_order
        self.n_pca_components = n_pca_components
        
        # Mean field coefficients and PCA basis
        self.mean_coefs_u = None
        self.mean_coefs_v = None
        self.pca_components = None
        self.pca_stds = None
        
        # Residual noise parameters
        self.residual_kappa = 15.0 # Von mises concentration
        self.parity_prob_180 = 0.5 # Probability of minutia pointing along reverse ridge direction

    def _build_poly_features(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Constructs normalized 2D polynomial feature matrix for coordinates x, y.
        """
        # Normalize to [-1, 1]
        x_norm = (x - 208.0) / 200.0
        y_norm = (y - 250.0) / 240.0
        
        features = []
        for i in range(self.poly_order + 1):
            for j in range(self.poly_order + 1 - i):
                features.append((x_norm**i) * (y_norm**j))
        return np.column_stack(features)

    def fit(self, fingerprints: list[dict]) -> "OrientationFieldModel":
        """
        Fits orientation fields from a list of pristine fingerprints.
        Each fingerprint is a dict with 'pts' ([N, 2]) and 'angles' ([N]).
        """
        all_coefs = []
        all_residuals = []
        
        for fp in fingerprints:
            pts = fp['pts']
            angles = fp['angles']
            if len(pts) < 10:
                continue
            
            # Doubled angle components
            u, v = to_doubled_angle_vector(angles)
            X = self._build_poly_features(pts[:, 0], pts[:, 1])
            
            # Ridge regression / least squares with small L2 regularization
            lambda_reg = 0.01 * np.eye(X.shape[1])
            XtX = X.T @ X + lambda_reg
            coef_u = np.linalg.solve(XtX, X.T @ u)
            coef_v = np.linalg.solve(XtX, X.T @ v)
            
            combined_coef = np.concatenate([coef_u, coef_v])
            all_coefs.append(combined_coef)
            
            # Evaluate field on training points to compute residual noise
            u_pred = X @ coef_u
            v_pred = X @ coef_v
            ridge_pred = from_doubled_angle_vector(u_pred, v_pred)
            
            # Compute residual difference modulo 180
            for a, r_pred in zip(angles, ridge_pred):
                # Minutia can point in ridge_pred or ridge_pred + 180
                diff1 = circular_distance_deg(a, r_pred)
                diff2 = circular_distance_deg(a, (r_pred + 180.0) % 360.0)
                best_diff = diff1 if abs(diff1) <= abs(diff2) else diff2
                all_residuals.append(best_diff)
                
        coef_matrix = np.array(all_coefs) # [N_fp, 2 * n_features]
        self.mean_coefs = np.mean(coef_matrix, axis=0)
        n_feat = self.mean_coefs.shape[0] // 2
        self.mean_coefs_u = self.mean_coefs[:n_feat]
        self.mean_coefs_v = self.mean_coefs[n_feat:]
        
        # Fit PCA on coefficients across fingerprints for diverse flow fields
        centered_coefs = coef_matrix - self.mean_coefs
        U, S, Vt = np.linalg.svd(centered_coefs, full_matrices=False)
        self.pca_components = Vt[:self.n_pca_components] # [k, 2 * n_features]
        self.pca_stds = S[:self.n_pca_components] / np.sqrt(len(coef_matrix) - 1)
        
        # Fit von Mises distribution to residuals
        res_arr = np.array(all_residuals)
        _, self.residual_kappa = fit_von_mises(res_arr)
        # Cap kappa to reasonable bounds to prevent over-dispersion or delta
        self.residual_kappa = float(np.clip(self.residual_kappa, 5.0, 50.0))
        
        return self

    def sample_identity_field_coefficients(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples a unique smooth orientation field for a new synthetic identity.
        """
        n_feat = len(self.mean_coefs_u)
        pca_weights = rng.normal(0.0, 1.0, size=self.n_pca_components) * self.pca_stds
        sample_coef = self.mean_coefs + (pca_weights @ self.pca_components)
        coef_u = sample_coef[:n_feat]
        coef_v = sample_coef[n_feat:]
        return coef_u, coef_v

    def assign_orientations(
        self,
        pts: np.ndarray,
        coef_u: np.ndarray,
        coef_v: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Given minutiae spatial positions and field coefficients, assigns realistic orientations.
        """
        n = len(pts)
        if n == 0:
            return np.array([], dtype=float)
            
        X = self._build_poly_features(pts[:, 0], pts[:, 1])
        u_pred = X @ coef_u
        v_pred = X @ coef_v
        
        # Base ridge angle in [0, 180)
        ridge_angles = from_doubled_angle_vector(u_pred, v_pred)
        
        # Choose direction parity: 0 deg or 180 deg along the ridge
        parity = rng.choice([0.0, 180.0], size=n, p=[1.0 - self.parity_prob_180, self.parity_prob_180])
        base_directions = (ridge_angles + parity) % 360.0
        
        # Sample circular residual noise from von Mises centered at 0
        noise = sample_von_mises(0.0, self.residual_kappa, size=n, rng=rng)
        # Center noise around 0 (since sample_von_mises returns [0, 360))
        noise = (noise + 180.0) % 360.0 - 180.0
        
        final_angles = (base_directions + noise) % 360.0
        # Quantize to 8-bit resolution (multiples of 360/256 = 1.40625 deg) matching real data
        bin_step = 360.0 / 256.0
        final_angles = np.round(final_angles / bin_step) * bin_step % 360.0
        
        return final_angles
