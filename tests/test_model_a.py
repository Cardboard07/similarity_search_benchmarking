"""
Unit tests for Model A identity generator.
"""
import numpy as np
import pytest
from src.models.model_a import ModelA
from src.statistics.spatial_stats import minimum_separation

@pytest.fixture
def dummy_model_a():
    # Fit on synthetic mock pristine fingerprints
    pristine_fps = []
    rng = np.random.default_rng(123)
    for ident in range(1, 30):
        n = int(rng.integers(30, 60))
        pts = np.column_stack([
            rng.uniform(30, 380, n),
            rng.uniform(30, 450, n)
        ])
        angles = rng.uniform(0, 360, n)
        types = rng.choice([1, 2], size=n, p=[0.4, 0.6])
        pristine_fps.append({
            'identity': ident,
            'pts': pts,
            'angles': angles,
            'types': types,
            'num_minutiae': n
        })
    model = ModelA(min_separation=8.0)
    model.fit(pristine_fps)
    return model

def test_model_a_generation_constraints(dummy_model_a):
    synth_id = dummy_model_a.generate_identity(seed=42)
    
    assert synth_id.num_minutiae >= 16
    assert len(synth_id.pts) == synth_id.num_minutiae
    assert len(synth_id.angles) == synth_id.num_minutiae
    assert len(synth_id.types) == synth_id.num_minutiae
    
    # Coordinate boundaries
    assert np.all(synth_id.pts[:, 0] >= 10.0)
    assert np.all(synth_id.pts[:, 0] <= 405.0)
    assert np.all(synth_id.pts[:, 1] >= 10.0)
    assert np.all(synth_id.pts[:, 1] <= 485.0)
    
    # Hard-core inhibition
    min_sep = minimum_separation(synth_id.pts)
    assert min_sep >= 7.9 # within numerical tolerance of 8.0
    
    # Minutiae angles valid circular range
    assert np.all(synth_id.angles >= 0.0)
    assert np.all(synth_id.angles < 360.0)
    
    # Minutiae types valid
    assert set(np.unique(synth_id.types)).issubset({1, 2})

def test_model_a_reproducibility(dummy_model_a):
    id1 = dummy_model_a.generate_identity(seed=999)
    id2 = dummy_model_a.generate_identity(seed=999)
    id3 = dummy_model_a.generate_identity(seed=1000)
    
    # Exact reproduction with same seed
    assert id1.num_minutiae == id2.num_minutiae
    np.testing.assert_array_equal(id1.pts, id2.pts)
    np.testing.assert_array_equal(id1.angles, id2.angles)
    np.testing.assert_array_equal(id1.types, id2.types)
    
    # Different seed yields different identity
    assert not np.array_equal(id1.pts, id3.pts)
