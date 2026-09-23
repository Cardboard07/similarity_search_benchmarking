"""
Unit tests for circular statistics and spatial point statistics.
"""
import numpy as np
import pytest
from src.statistics.circular_stats import (
    circular_distance_deg,
    circular_abs_distance_deg,
    ridge_flow_distance_deg,
    circular_mean_deg,
    circular_variance,
    fit_von_mises,
    sample_von_mises
)
from src.statistics.spatial_stats import (
    nearest_neighbor_distances,
    minimum_separation,
    local_densities,
    boundary_distances
)

def test_circular_distance_periodicity():
    # 359 deg and 1 deg should be 2 degrees apart
    diff = circular_abs_distance_deg(359.0, 1.0)
    assert pytest.approx(diff, abs=1e-5) == 2.0
    
    # 1 deg and 359 deg signed difference
    diff_signed = circular_distance_deg(1.0, 359.0)
    assert pytest.approx(diff_signed, abs=1e-5) == 2.0
    
    # Opposite angle
    assert pytest.approx(circular_abs_distance_deg(0.0, 180.0), abs=1e-5) == 180.0

def test_ridge_flow_symmetry():
    # In ridge flow, theta and theta + 180 represent the exact same line/ridge
    assert pytest.approx(ridge_flow_distance_deg(10.0, 190.0), abs=1e-5) == 0.0
    assert pytest.approx(ridge_flow_distance_deg(175.0, 5.0), abs=1e-5) == -10.0

def test_circular_mean_and_variance():
    # Angles clustered around 0 deg (350 deg and 10 deg)
    angles = np.array([350.0, 355.0, 0.0, 5.0, 10.0])
    mean_ang = circular_mean_deg(angles)
    assert (mean_ang < 5.0 or mean_ang > 355.0) # Near 0
    var = circular_variance(angles)
    assert var < 0.1 # Very low circular variance

def test_von_mises_sampling():
    rng = np.random.default_rng(42)
    samples = sample_von_mises(mu_deg=90.0, kappa=20.0, size=500, rng=rng)
    assert len(samples) == 500
    assert np.all(samples >= 0.0) and np.all(samples < 360.0)
    mu_est, kappa_est = fit_von_mises(samples)
    assert pytest.approx(mu_est, abs=5.0) == 90.0
    assert pytest.approx(kappa_est, abs=8.0) == 20.0

def test_spatial_statistics():
    pts = np.array([
        [100.0, 100.0],
        [100.0, 110.0],
        [100.0, 125.0],
        [200.0, 200.0]
    ])
    nn = nearest_neighbor_distances(pts, k=1)
    assert pytest.approx(nn[0], abs=1e-5) == 10.0
    assert pytest.approx(minimum_separation(pts), abs=1e-5) == 10.0
    
    dens = local_densities(pts, radius=15.0)
    assert dens[0] == 1.0 # only (100, 110) within 15px
    assert dens[1] == 2.0 # (100, 100) and (100, 125) within 15px

def test_boundary_distances():
    pts = np.array([
        [15.0, 20.0],
        [400.0, 500.0]
    ])
    bd = boundary_distances(pts, width=416.0, height=560.0)
    assert pytest.approx(bd[0], abs=1e-5) == 15.0 # closest to left (x=0)
    assert pytest.approx(bd[1], abs=1e-5) == 16.0 # closest to right (416-400=16)
