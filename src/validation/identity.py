"""
Identity preservation validation module.
Verifies that all 15 observations of one synthetic identity maintain strong geometric correspondence
and distinctiveness compared to unrelated identities (intra-class vs inter-class matching).
"""
import numpy as np
from src.models.correspondence import MinutiaMatcher

class IdentityPreservationValidator:
    """
    Evaluates identity consistency across orientations and quality levels.
    """
    def __init__(self, spatial_gate: float = 18.0, angular_gate_deg: float = 40.0):
        self.matcher = MinutiaMatcher(
            spatial_gate=spatial_gate,
            angular_gate_deg=angular_gate_deg,
            refine_iterations=2
        )

    def evaluate_identity_observations(
        self,
        pristine_obs: dict,
        other_observations: list[dict]
    ) -> list[dict]:
        """
        Matches all observations of an identity against its reference pristine observation (ori=1, q=3).
        """
        results = []
        p_pts = pristine_obs['pts']
        p_ang = pristine_obs['angles']
        p_typ = pristine_obs['types']

        for obs in other_observations:
            res = self.matcher.match(
                pts_a=p_pts,
                angles_a=p_ang,
                types_a=p_typ,
                pts_b=obs['pts'],
                angles_b=obs['angles'],
                types_b=obs['types']
            )
            results.append({
                'orientation': obs['orientation'],
                'quality': obs['quality'],
                'match_rate': res.survival_rate,
                'mean_spatial_error': float(np.mean(res.spatial_distances)) if len(res.spatial_distances) > 0 else 0.0,
                'mean_angular_error': float(np.mean(res.angular_differences)) if len(res.angular_differences) > 0 else 0.0,
                'false_count': res.false_minutiae_count
            })
            
        return results

    def compute_intra_vs_inter_class_scores(
        self,
        synthetic_identities_obs: list[list[dict]],
        num_inter_comparisons: int = 200
    ) -> dict:
        """
        Compares match score distributions for intra-identity pairs vs inter-identity pairs.
        """
        intra_scores = []
        for id_obs in synthetic_identities_obs:
            ref = id_obs[0] # pristine (ori=1, q=3)
            for obs in id_obs[1:]:
                res = self.matcher.match(
                    pts_a=ref['pts'],
                    angles_a=ref['angles'],
                    types_a=ref['types'],
                    pts_b=obs['pts'],
                    angles_b=obs['angles'],
                    types_b=obs['types']
                )
                intra_scores.append(res.survival_rate)
                
        inter_scores = []
        n_ids = len(synthetic_identities_obs)
        if n_ids > 1:
            for _ in range(num_inter_comparisons):
                i, j = np.random.choice(n_ids, size=2, replace=False)
                ref_i = synthetic_identities_obs[i][0]
                obs_j = synthetic_identities_obs[j][np.random.randint(len(synthetic_identities_obs[j]))]
                res = self.matcher.match(
                    pts_a=ref_i['pts'],
                    angles_a=ref_i['angles'],
                    types_a=ref_i['types'],
                    pts_b=obs_j['pts'],
                    angles_b=obs_j['angles'],
                    types_b=obs_j['types']
                )
                inter_scores.append(res.survival_rate)

        intra_arr = np.array(intra_scores)
        inter_arr = np.array(inter_scores) if len(inter_scores) > 0 else np.array([0.0])

        return {
            'intra_match_rate_mean': float(np.mean(intra_arr)),
            'intra_match_rate_std': float(np.std(intra_arr)),
            'intra_match_rate_min': float(np.min(intra_arr)),
            'inter_match_rate_mean': float(np.mean(inter_arr)),
            'inter_match_rate_std': float(np.std(inter_arr)),
            'inter_match_rate_max': float(np.max(inter_arr)),
            'd_prime_separability': float((np.mean(intra_arr) - np.mean(inter_arr)) / np.sqrt(0.5 * (np.var(intra_arr) + np.var(inter_arr) + 1e-6)))
        }
