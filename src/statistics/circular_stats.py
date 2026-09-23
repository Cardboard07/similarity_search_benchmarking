"""
Circular statistics utilities for fingerprint minutiae orientations and ridge flow fields.
"""
import numpy as np
from scipy import special

def deg_to_rad(deg: np.ndarray) -> np.ndarray:
    return np.radians(deg)

def rad_to_deg(rad: np.ndarray) -> np.ndarray:
    return np.degrees(rad) % 360.0

def circular_distance_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Computes signed circular difference angle1 - angle2 in [-180, 180] degrees.
    """
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)
    diff_rad = np.arctan2(np.sin(rad1 - rad2), np.cos(rad1 - rad2))
    return np.degrees(diff_rad)

def circular_abs_distance_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Computes absolute circular distance in [0, 180] degrees.
    """
    return np.abs(circular_distance_deg(angle1, angle2))

def ridge_flow_distance_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Computes nematic ridge flow difference (modulo 180 degrees) in [-90, 90] degrees.
    """
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)
    diff_rad = 0.5 * np.arctan2(np.sin(2.0 * (rad1 - rad2)), np.cos(2.0 * (rad1 - rad2)))
    return np.degrees(diff_rad)

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """
    Computes circular mean of angles in degrees.
    """
    rads = np.radians(angles_deg)
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    return float(np.degrees(mean_rad) % 360.0)

def circular_variance(angles_deg: np.ndarray) -> float:
    """
    Computes sample circular variance in [0, 1].
    Var = 1 - R, where R is resultant vector length.
    """
    rads = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(rads))
    cos_mean = np.mean(np.cos(rads))
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    return float(1.0 - r)

def circular_std_deg(angles_deg: np.ndarray) -> float:
    """
    Computes circular standard deviation: sqrt(-2 * ln(R)) in degrees.
    """
    rads = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(rads))
    cos_mean = np.mean(np.cos(rads))
    r = np.clip(np.sqrt(sin_mean**2 + cos_mean**2), 1e-12, 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))

def fit_von_mises(angles_deg: np.ndarray) -> tuple[float, float]:
    """
    Maximum Likelihood Estimation of Von Mises parameters (mu in degrees, kappa concentration).
    """
    rads = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(rads))
    cos_mean = np.mean(np.cos(rads))
    r_bar = np.clip(np.sqrt(sin_mean**2 + cos_mean**2), 1e-6, 0.999999)
    mu_rad = np.arctan2(sin_mean, cos_mean)
    mu_deg = float(np.degrees(mu_rad) % 360.0)
    
    # Best-Fisher approximation for kappa
    if r_bar < 0.53:
        kappa = 2.0 * r_bar + r_bar**3 + (5.0 / 6.0) * r_bar**5
    elif r_bar < 0.85:
        kappa = -0.4 + 1.39 * r_bar + 0.43 / (1.0 - r_bar)
    else:
        kappa = 1.0 / (r_bar**3 - 4.0 * r_bar**2 + 3.0 * r_bar)
    
    return mu_deg, float(kappa)

def sample_von_mises(mu_deg: float, kappa: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Samples from Von Mises distribution with mean mu_deg and concentration kappa.
    """
    mu_rad = np.radians(mu_deg)
    samples_rad = rng.vonmises(mu_rad, kappa, size=size)
    return np.degrees(samples_rad) % 360.0

def to_doubled_angle_vector(angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts angles to 2D doubled-angle vector representation: (cos(2*theta), sin(2*theta)).
    """
    rads = np.radians(angles_deg)
    return np.cos(2.0 * rads), np.sin(2.0 * rads)

def from_doubled_angle_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Converts 2D doubled-angle vector representation back to ridge angle in [0, 180) degrees.
    """
    rads = 0.5 * np.arctan2(v, u)
    return np.degrees(rads) % 180.0
