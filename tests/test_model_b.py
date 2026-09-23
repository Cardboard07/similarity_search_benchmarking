"""
Unit tests for Model B (AngleModel & QualityModel) and identity preservation.
"""
import numpy as np
import pytest
from src.models.angle_model import AngleModel
from src.models.quality_model import QualityModel
from src.models.correspondence import MinutiaMatcher

@pytest.fixture
def mock_pristine_fingerprint():
    rng = np.random.default_rng(101)
    n = 50
    pts = np.column_stack([
        rng.uniform(50, 350, n),
        rng.uniform(50, 420, n)
    ])
    angles = rng.uniform(0, 360, n)
    types = rng.choice([1, 2], size=n, p=[0.4, 0.6])
    return pts, angles, types

def test_angle_model_impressions(mock_pristine_fingerprint):
    pts, angles, types = mock_pristine_fingerprint
    angle_model = AngleModel()
    
    # Orientation 1: identity
    rng = np.random.default_rng(42)
    p1, a1, t1, param1 = angle_model.apply_impression(pts, angles, types, orientation_idx=1, rng=rng)
    np.testing.assert_array_equal(p1, pts)
    np.testing.assert_array_equal(a1, angles)
    
    # Orientation 2 & 3: realistic deformation
    p2, a2, t2, param2 = angle_model.apply_impression(pts, angles, types, orientation_idx=2, rng=rng)
    assert not np.array_equal(p2, pts)
    # Check bounded displacement
    dists = np.linalg.norm(p2 - pts, axis=1)
    assert np.mean(dists) < 15.0 # realistic displacement within 15px
    assert np.all(p2[:, 0] >= 10.0) and np.all(p2[:, 0] <= 405.0)

def test_quality_model_degradation(mock_pristine_fingerprint):
    pts, angles, types = mock_pristine_fingerprint
    quality_model = QualityModel()
    matcher = MinutiaMatcher(spatial_gate=18.0)
    
    rng = np.random.default_rng(42)
    # Quality 3 (pristine) -> exact match
    p3, a3, t3, q3 = quality_model.apply_quality_degradation(pts, angles, types, quality_level=3, rng=rng)
    np.testing.assert_array_equal(p3, pts)
    
    # Check progressive degradation from good (7) to noisy (5) to bad (1) to very bad (9)
    res_list = []
    for q in [7, 5, 1, 9]:
        pq, aq, tq, qq = quality_model.apply_quality_degradation(pts, angles, types, quality_level=q, rng=rng)
        res = matcher.match(pts_a=pts, angles_a=angles, types_a=types, pts_b=pq, angles_b=aq, types_b=tq)
        res_list.append((q, res.survival_rate, np.mean(res.spatial_distances)))

    # High survival rate across all levels (> 75%) preserving identity
    for q, surv, err in res_list:
        assert surv >= 0.75
        assert err < 10.0
