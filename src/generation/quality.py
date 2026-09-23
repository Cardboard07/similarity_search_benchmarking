"""
Quality degradation generation routines (qualities 3, 7, 5, 1, 9).
"""
import numpy as np
from src.models.quality_model import QualityModel

def generate_quality_observations(
    impression_data: dict,
    quality_model: QualityModel,
    base_seed: int
) -> list[dict]:
    """
    Produces 5 quality observations for a single impression:
    q3: pristine
    q7: good
    q5: noisy
    q1: bad
    q9: very bad
    Returns a list of 5 observation dicts.
    """
    observations = []
    pts = impression_data['pts']
    angles = impression_data['angles']
    types = impression_data['types']
    ori = impression_data['orientation']
    
    # Standard quality ordering
    for q in [3, 7, 5, 1, 9]:
        rng = np.random.default_rng(base_seed + q * 1013)
        deg_pts, deg_angles, deg_types, min_qualities = quality_model.apply_quality_degradation(
            pts=pts,
            angles=angles,
            types=types,
            quality_level=q,
            rng=rng
        )
        observations.append({
            'orientation': ori,
            'quality': q,
            'pts': deg_pts,
            'angles': deg_angles,
            'types': deg_types,
            'qualities': min_qualities,
            'num_minutiae': len(deg_pts)
        })
        
    return observations
