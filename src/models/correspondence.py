"""
Minutia correspondence algorithm for paired impressions and quality levels.
Implements robust bipartite matching with circular angular cost and outlier gating.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from src.statistics.circular_stats import circular_abs_distance_deg

@dataclass
class CorrespondenceResult:
    matched_a_indices: np.ndarray
    matched_b_indices: np.ndarray
    disappeared_a_indices: np.ndarray
    spurious_b_indices: np.ndarray
    spatial_distances: np.ndarray
    angular_differences: np.ndarray
    type_matches: np.ndarray
    estimated_tx: float
    estimated_ty: float
    estimated_rotation_deg: float
    estimated_scale: float
    survival_rate: float
    false_minutiae_count: int

class MinutiaMatcher:
    """
    Minutia correspondence matcher.
    Calculates optimal matching between two sets of minutiae accounting for translation,
    rotation, spatial distance, circular angle difference, and type compatibility.
    """
    def __init__(
        self,
        spatial_gate: float = 15.0,
        angular_gate_deg: float = 35.0,
        weight_spatial: float = 1.0,
        weight_angle: float = 0.2, # ~0.2 * 10 deg = 2.0 cost
        weight_type_mismatch: float = 10.0,
        refine_iterations: int = 2
    ):
        self.spatial_gate = spatial_gate
        self.angular_gate_deg = angular_gate_deg
        self.weight_spatial = weight_spatial
        self.weight_angle = weight_angle
        self.weight_type_mismatch = weight_type_mismatch
        self.refine_iterations = refine_iterations

    def match(
        self,
        pts_a: np.ndarray,
        angles_a: np.ndarray,
        types_a: np.ndarray,
        pts_b: np.ndarray,
        angles_b: np.ndarray,
        types_b: np.ndarray
    ) -> CorrespondenceResult:
        """
        Solves correspondence from set A (source, e.g. pristine) to set B (target, e.g. degraded/transformed).
        """
        n_a = len(pts_a)
        n_b = len(pts_b)
        
        if n_a == 0 or n_b == 0:
            return CorrespondenceResult(
                matched_a_indices=np.array([], dtype=int),
                matched_b_indices=np.array([], dtype=int),
                disappeared_a_indices=np.arange(n_a),
                spurious_b_indices=np.arange(n_b),
                spatial_distances=np.array([], dtype=float),
                angular_differences=np.array([], dtype=float),
                type_matches=np.array([], dtype=bool),
                estimated_tx=0.0,
                estimated_ty=0.0,
                estimated_rotation_deg=0.0,
                estimated_scale=1.0,
                survival_rate=0.0,
                false_minutiae_count=n_b
            )

        # Initial coarse alignment (translation via centroids)
        current_tx = float(np.mean(pts_b[:, 0]) - np.mean(pts_a[:, 0]))
        current_ty = float(np.mean(pts_b[:, 1]) - np.mean(pts_a[:, 1]))
        current_rot = 0.0
        current_scale = 1.0

        pts_a_aligned = pts_a.copy()
        
        # Iterative matching and alignment refinement
        for it in range(self.refine_iterations):
            # Transform A towards B
            c_a = np.mean(pts_a, axis=0)
            p_centered = pts_a - c_a
            rad = np.radians(current_rot)
            cos_r, sin_r = np.cos(rad), np.sin(rad)
            R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            pts_a_aligned = current_scale * (p_centered @ R.T) + c_a + np.array([current_tx, current_ty])
            angles_a_aligned = (angles_a + current_rot) % 360.0

            # Compute cost matrix
            dist_spatial = cdist(pts_a_aligned, pts_b) # [N_a, N_b]
            
            # Pairwise circular angular difference
            ang_diff = np.zeros((n_a, n_b), dtype=float)
            for i in range(n_a):
                ang_diff[i, :] = circular_abs_distance_deg(angles_a_aligned[i], angles_b)
                
            # Type mismatch matrix
            type_mismatch = (types_a[:, None] != types_b[None, :]).astype(float)
            
            cost_matrix = (
                self.weight_spatial * dist_spatial +
                self.weight_angle * ang_diff +
                self.weight_type_mismatch * type_mismatch
            )
            
            # Penalize pairs exceeding gates so Hungarian doesn't force bad pairs
            gate_mask = (dist_spatial > self.spatial_gate) | (ang_diff > self.angular_gate_deg)
            cost_matrix[gate_mask] += 1000.0

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Filter matches satisfying hard gates
            valid_mask = (dist_spatial[row_ind, col_ind] <= self.spatial_gate) & (ang_diff[row_ind, col_ind] <= self.angular_gate_deg)
            matched_a = row_ind[valid_mask]
            matched_b = col_ind[valid_mask]

            if len(matched_a) >= 4 and it < self.refine_iterations - 1:
                # Refine rigid alignment using matched pairs via Procrustes/SVD
                p1 = pts_a[matched_a]
                p2 = pts_b[matched_b]
                c1 = p1.mean(axis=0)
                c2 = p2.mean(axis=0)
                p1_c = p1 - c1
                p2_c = p2 - c2
                
                H = p1_c.T @ p2_c
                U, S, Vt = np.linalg.svd(H)
                R_mat = Vt.T @ U.T
                if np.linalg.det(R_mat) < 0:
                    Vt[-1, :] *= -1
                    R_mat = Vt.T @ U.T
                current_rot = float(np.degrees(np.arctan2(R_mat[1, 0], R_mat[0, 0])))
                current_scale = float(np.sum(S) / (np.sum(p1_c**2) + 1e-8))
                current_tx = float(c2[0] - c1[0])
                current_ty = float(c2[1] - c1[1])

        # Final matched characteristics
        matched_spatial_dists = np.linalg.norm(pts_a[matched_a] + np.array([current_tx, current_ty]) - pts_b[matched_b], axis=1) if len(matched_a) > 0 else np.array([], dtype=float)
        matched_ang_diffs = circular_abs_distance_deg(angles_a[matched_a] + current_rot, angles_b[matched_b]) if len(matched_a) > 0 else np.array([], dtype=float)
        type_matches = (types_a[matched_a] == types_b[matched_b]) if len(matched_a) > 0 else np.array([], dtype=bool)

        matched_a_set = set(matched_a)
        matched_b_set = set(matched_b)
        disappeared_a = np.array([i for i in range(n_a) if i not in matched_a_set], dtype=int)
        spurious_b = np.array([j for j in range(n_b) if j not in matched_b_set], dtype=int)

        survival_rate = len(matched_a) / float(n_a) if n_a > 0 else 0.0
        false_count = len(spurious_b)

        return CorrespondenceResult(
            matched_a_indices=matched_a,
            matched_b_indices=matched_b,
            disappeared_a_indices=disappeared_a,
            spurious_b_indices=spurious_b,
            spatial_distances=matched_spatial_dists,
            angular_differences=matched_ang_diffs,
            type_matches=type_matches,
            estimated_tx=current_tx,
            estimated_ty=current_ty,
            estimated_rotation_deg=current_rot,
            estimated_scale=current_scale,
            survival_rate=survival_rate,
            false_minutiae_count=false_count
        )
