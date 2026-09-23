"""
Impression / orientation generation routines (orientations 1, 2, 3).
"""
import numpy as np
from src.models.angle_model import AngleModel, TransformationParameters
from src.models.model_a import SyntheticIdentity

def generate_impressions(
    synthetic_identity: SyntheticIdentity,
    angle_model: AngleModel,
    base_seed: int
) -> list[dict]:
    """
    Produces 3 impressions (orientations 1, 2, 3) for a given synthetic pristine identity.
    Returns a list of 3 dicts containing 'orientation', 'pts', 'angles', 'types', 'transform_params'.
    """
    impressions = []
    
    for ori in [1, 2, 3]:
        rng = np.random.default_rng(base_seed + ori * 10007)
        pts, angles, types, params = angle_model.apply_impression(
            pts=synthetic_identity.pts,
            angles=synthetic_identity.angles,
            types=synthetic_identity.types,
            orientation_idx=ori,
            rng=rng
        )
        impressions.append({
            'orientation': ori,
            'pts': pts,
            'angles': angles,
            'types': types,
            'transform_params': params
        })
        
    return impressions
