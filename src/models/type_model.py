"""
Minutia type model (Type 1: termination/ridge ending, Type 2: bifurcation).
Evaluates conditional dependencies P(type | x, y, density) and uses conditional model
if statistically justified, otherwise using calibrated empirical global distribution.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.statistics.spatial_stats import local_densities

class MinutiaTypeModel:
    """
    Assigns minutia types (1 or 2) conditioned on spatial location and local density.
    """
    def __init__(self, method: str = "spatial_conditional"):
        self.method = method
        self.global_p_type1 = 0.3976 # Real data calibrated default
        self.model = None
        self.use_conditional = False

    def fit(self, fingerprints: list[dict]) -> "MinutiaTypeModel":
        all_features = []
        all_labels = [] # 1 -> 0, 2 -> 1
        
        for fp in fingerprints:
            pts = fp['pts']
            types = fp['types']
            if len(pts) == 0:
                continue
            densities = local_densities(pts, radius=30.0)
            
            # Features: normalized x, y, and density
            x_norm = (pts[:, 0] - 208.0) / 200.0
            y_norm = (pts[:, 1] - 250.0) / 240.0
            dens_norm = (densities - 2.5) / 2.0
            
            feats = np.column_stack([x_norm, y_norm, dens_norm])
            labels = (types == 2).astype(int)
            
            all_features.append(feats)
            all_labels.append(labels)
            
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # Empirical global ratio
        self.global_p_type1 = float(np.mean(y == 0))
        
        if self.method == "spatial_conditional":
            # Fit regularized logistic regression
            lr = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            lr.fit(X, y)
            
            # Check likelihood improvement
            log_loss_base = -np.mean(y * np.log(1.0 - self.global_p_type1) + (1.0 - y) * np.log(self.global_p_type1))
            probs_pred = lr.predict_proba(X)[:, 1]
            probs_pred = np.clip(probs_pred, 1e-6, 1.0 - 1e-6)
            log_loss_lr = -np.mean(y * np.log(probs_pred) + (1.0 - y) * np.log(1.0 - probs_pred))
            
            # Use conditional if it improves log loss
            if log_loss_lr < log_loss_base - 0.002:
                self.model = lr
                self.use_conditional = True
            else:
                self.use_conditional = False
        else:
            self.use_conditional = False
            
        return self

    def assign_types(self, pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Assigns minutia types (1 or 2) to an array of points [N, 2].
        """
        n = len(pts)
        if n == 0:
            return np.array([], dtype=int)
            
        if self.use_conditional and self.model is not None:
            densities = local_densities(pts, radius=30.0)
            x_norm = (pts[:, 0] - 208.0) / 200.0
            y_norm = (pts[:, 1] - 250.0) / 240.0
            dens_norm = (densities - 2.5) / 2.0
            feats = np.column_stack([x_norm, y_norm, dens_norm])
            
            # Probability of type 2
            p_type2 = self.model.predict_proba(feats)[:, 1]
            u = rng.uniform(0.0, 1.0, size=n)
            types = np.where(u < p_type2, 2, 1)
        else:
            p_type1 = self.global_p_type1
            u = rng.uniform(0.0, 1.0, size=n)
            types = np.where(u < p_type1, 1, 2)
            
        return types.astype(int)
