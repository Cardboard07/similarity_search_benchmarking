"""
Pristine fingerprint identity generation routines.
"""
import numpy as np
from src.models.model_a import ModelA, SyntheticIdentity

def generate_pristine_identity(model_a: ModelA, identity_seed: int) -> SyntheticIdentity:
    """
    Generates a single pristine synthetic identity from Model A using an explicit seed.
    """
    return model_a.generate_identity(identity_seed)
