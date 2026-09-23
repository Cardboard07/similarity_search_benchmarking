"""
Novelty and anti-memorization validation module.
Verifies that generated synthetic identities are not reproductions, copies, or slight perturbations
of any existing real fingerprint identity in the training population.
"""
from dataclasses import dataclass
import numpy as np
from src.models.correspondence import MinutiaMatcher

@dataclass
class NoveltyResult:
    synthetic_identity: int
    nearest_real_identity: int
    max_match_similarity: float
    mean_spatial_error: float
    is_novel: bool

class NoveltyValidator:
    """
    Evaluates novelty of synthetic identities by matching against real pristine fingerprints.
    """
    def __init__(
        self,
        real_pristine_fingerprints: list[dict],
        copy_threshold: float = 0.65,
        spatial_gate: float = 12.0
    ):
        self.real_fps = real_pristine_fingerprints
        self.copy_threshold = copy_threshold
        self.matcher = MinutiaMatcher(
            spatial_gate=spatial_gate,
            angular_gate_deg=30.0,
            refine_iterations=2
        )
        
        # Precompute real fingerprints summary descriptors for fast filtering
        self.real_counts = np.array([len(fp['pts']) for fp in self.real_fps])
        self.real_centroids = np.array([np.mean(fp['pts'], axis=0) if len(fp['pts']) > 0 else [0.0, 0.0] for fp in self.real_fps])

    def check_identity_novelty(
        self,
        synthetic_identity_id: int,
        synth_pts: np.ndarray,
        synth_angles: np.ndarray,
        synth_types: np.ndarray,
        top_k_candidates: int = 40
    ) -> NoveltyResult:
        """
        Computes maximum similarity between a synthetic identity and all real fingerprints.
        Uses descriptor pre-filtering + full geometric correspondence on top candidates.
        """
        synth_count = len(synth_pts)
        synth_centroid = np.mean(synth_pts, axis=0) if synth_count > 0 else np.array([0.0, 0.0])
        
        # Filter candidate real fingerprints by count and centroid proximity
        count_diffs = np.abs(self.real_counts - synth_count)
        centroid_dists = np.linalg.norm(self.real_centroids - synth_centroid, axis=1)
        # Combined descriptor distance
        ranking_scores = count_diffs + 0.1 * centroid_dists
        candidate_indices = np.argsort(ranking_scores)[:top_k_candidates]

        best_similarity = 0.0
        best_real_id = -1
        best_spatial_err = 0.0

        for idx in candidate_indices:
            real_fp = self.real_fps[idx]
            res = self.matcher.match(
                pts_a=synth_pts,
                angles_a=synth_angles,
                types_a=synth_types,
                pts_b=real_fp['pts'],
                angles_b=real_fp['angles'],
                types_b=real_fp['types']
            )
            sim = res.survival_rate
            if sim > best_similarity:
                best_similarity = sim
                best_real_id = real_fp.get('identity', idx + 1)
                best_spatial_err = float(np.mean(res.spatial_distances)) if len(res.spatial_distances) > 0 else 0.0

        is_novel = (best_similarity < self.copy_threshold)
        
        return NoveltyResult(
            synthetic_identity=synthetic_identity_id,
            nearest_real_identity=int(best_real_id),
            max_match_similarity=float(best_similarity),
            mean_spatial_error=float(best_spatial_err),
            is_novel=is_novel
        )

    def evaluate_batch_novelty(self, synthetic_identities: list[dict]) -> dict:
        """
        Evaluates novelty for a batch of synthetic pristine identities.
        """
        results = []
        for s in synthetic_identities:
            res = self.check_identity_novelty(
                synthetic_identity_id=s['identity'],
                synth_pts=s['pts'],
                synth_angles=s['angles'],
                synth_types=s['types']
            )
            results.append(res)

        similarities = [r.max_match_similarity for r in results]
        novel_flags = [r.is_novel for r in results]

        return {
            'num_evaluated': len(results),
            'all_novel': all(novel_flags),
            'novel_percentage': float(np.mean(novel_flags) * 100.0),
            'max_similarity_observed': float(np.max(similarities)) if similarities else 0.0,
            'mean_similarity_to_nearest': float(np.mean(similarities)) if similarities else 0.0,
            'copy_threshold': self.copy_threshold
        }
