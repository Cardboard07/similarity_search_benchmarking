"""
Unit tests for minutia correspondence matcher.
"""
import numpy as np
import pytest
from src.models.correspondence import MinutiaMatcher

def test_correspondence_perfect_match():
    matcher = MinutiaMatcher()
    pts = np.array([[100.0, 150.0], [200.0, 250.0], [300.0, 350.0]])
    angles = np.array([45.0, 90.0, 180.0])
    types = np.array([1, 2, 1])
    
    res = matcher.match(pts, angles, types, pts, angles, types)
    assert res.survival_rate == 1.0
    assert len(res.matched_a_indices) == 3
    assert len(res.disappeared_a_indices) == 0
    assert res.false_minutiae_count == 0
    assert np.all(res.spatial_distances < 1e-4)

def test_correspondence_shuffled_order_and_noise():
    matcher = MinutiaMatcher(spatial_gate=15.0)
    rng = np.random.default_rng(42)
    n = 20
    pts_a = rng.uniform(50, 350, size=(n, 2))
    angles_a = rng.uniform(0, 360, size=n)
    types_a = rng.choice([1, 2], size=n)

    # Shuffle order, add small jitter (2 px, 2 deg)
    perm = rng.permutation(n)
    pts_b = pts_a[perm] + rng.normal(0, 1.5, size=(n, 2))
    angles_b = (angles_a[perm] + rng.normal(0, 1.5, size=n)) % 360.0
    types_b = types_a[perm]

    # Add 2 spurious minutiae to set B
    false_pts = np.array([[40.0, 40.0], [380.0, 480.0]])
    pts_b = np.vstack([pts_b, false_pts])
    angles_b = np.concatenate([angles_b, [0.0, 180.0]])
    types_b = np.concatenate([types_b, [1, 2]])

    res = matcher.match(pts_a, angles_a, types_a, pts_b, angles_b, types_b)
    assert res.survival_rate >= 0.90 # At least 90% correctly matched despite shuffling
    assert res.false_minutiae_count >= 2 # Spurious detected
