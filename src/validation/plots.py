"""
Diagnostic plotting utilities comparing real and synthetic fingerprint minutiae.
Produces publication-quality figures: count distributions, spatial NN distributions,
circular angle distributions, pairwise distances, orientation flow fields, and degradation curves.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_count_distribution(
    real_counts: np.ndarray,
    synth_counts: np.ndarray,
    output_path: str
):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(15, 100, 35)
    plt.hist(real_counts, bins=bins, density=True, alpha=0.55, color='#1f77b4', label=f'Real Pristine (mean={np.mean(real_counts):.1f})', edgecolor='black')
    plt.hist(synth_counts, bins=bins, density=True, alpha=0.55, color='#ff7f0e', label=f'Synthetic (mean={np.mean(synth_counts):.1f})', edgecolor='black')
    plt.title('Minutiae Count Distribution: Real vs Synthetic', fontsize=13, fontweight='bold')
    plt.xlabel('Minutiae Count (N)', fontsize=11)
    plt.ylabel('Probability Density', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_spatial_nn_distribution(
    real_nn_dists: np.ndarray,
    synth_nn_dists: np.ndarray,
    output_path: str
):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 60, 40)
    plt.hist(real_nn_dists, bins=bins, density=True, alpha=0.55, color='#2ca02c', label=f'Real (mean={np.mean(real_nn_dists):.1f}px)', edgecolor='black')
    plt.hist(synth_nn_dists, bins=bins, density=True, alpha=0.55, color='#d62728', label=f'Synthetic (mean={np.mean(synth_nn_dists):.1f}px)', edgecolor='black')
    plt.title('Nearest Neighbor Distance (1-NN): Real vs Synthetic', fontsize=13, fontweight='bold')
    plt.xlabel('Distance (pixels)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_pairwise_distribution(
    real_pw_dists: np.ndarray,
    synth_pw_dists: np.ndarray,
    output_path: str
):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 400, 50)
    plt.hist(real_pw_dists, bins=bins, density=True, alpha=0.55, color='#9467bd', label=f'Real (mean={np.mean(real_pw_dists):.1f}px)', edgecolor='black')
    plt.hist(synth_pw_dists, bins=bins, density=True, alpha=0.55, color='#8c564b', label=f'Synthetic (mean={np.mean(synth_pw_dists):.1f}px)', edgecolor='black')
    plt.title('Pairwise Distance Distribution: Real vs Synthetic', fontsize=13, fontweight='bold')
    plt.xlabel('Pairwise Distance (pixels)', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_circular_polar_distribution(
    real_angles_deg: np.ndarray,
    synth_angles_deg: np.ndarray,
    output_path: str
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    n_bins = 36
    bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    r_rad = np.radians(real_angles_deg % 360.0)
    s_rad = np.radians(synth_angles_deg % 360.0)
    
    hr, _ = np.histogram(r_rad, bins=bin_edges, density=True)
    hs, _ = np.histogram(s_rad, bins=bin_edges, density=True)
    
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (2.0 * np.pi) / n_bins
    
    ax.bar(centers, hr, width=width, bottom=0.0, alpha=0.5, color='blue', label='Real Pristine', edgecolor='black')
    ax.bar(centers, hs, width=width, bottom=0.0, alpha=0.5, color='orange', label='Synthetic', edgecolor='black')
    
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title('Minutiae Orientation Distribution (Polar)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower left', bbox_to_anchor=(0.85, 0.9))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_synthetic_fingerprint_sample(
    pts: np.ndarray,
    angles: np.ndarray,
    types: np.ndarray,
    title: str,
    output_path: str
):
    plt.figure(figsize=(6, 8))
    # Image frame 416 x 560
    plt.xlim(0, 416)
    plt.ylim(560, 0) # Invert Y to match image coordinates
    
    # Minutiae points colored by type
    mask1 = (types == 1)
    mask2 = (types == 2)
    
    rads = np.radians(angles)
    u = np.cos(rads) * 16.0
    v = np.sin(rads) * 16.0
    
    if np.any(mask1):
        plt.scatter(pts[mask1, 0], pts[mask1, 1], c='crimson', s=45, label='Type 1 (Ending)', zorder=3, edgecolors='black')
        plt.quiver(pts[mask1, 0], pts[mask1, 1], u[mask1], v[mask1], color='crimson', angles='xy', scale_units='xy', scale=1.0, width=0.005, headwidth=4, zorder=2)
    if np.any(mask2):
        plt.scatter(pts[mask2, 0], pts[mask2, 1], c='navy', s=45, label='Type 2 (Bifurcation)', zorder=3, edgecolors='black')
        plt.quiver(pts[mask2, 0], pts[mask2, 1], u[mask2], v[mask2], color='navy', angles='xy', scale_units='xy', scale=1.0, width=0.005, headwidth=4, zorder=2)
        
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_quality_degradation_curves(
    real_quality_stats: dict[int, dict],
    synth_quality_stats: dict[int, dict],
    output_path: str
):
    qualities = [3, 7, 5, 1, 9]
    q_labels = ['q3\n(Pristine)', 'q7\n(Good)', 'q5\n(Noisy)', 'q1\n(Bad)', 'q9\n(Very Bad)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Survival Rate
    r_surv = [100.0] + [real_quality_stats[q]['surv'] * 100.0 for q in [7, 5, 1, 9]]
    s_surv = [100.0] + [synth_quality_stats[q]['surv'] * 100.0 for q in [7, 5, 1, 9]]
    axes[0].plot(range(5), r_surv, 'o-', color='#1f77b4', label='Real Data', linewidth=2)
    axes[0].plot(range(5), s_surv, 's--', color='#ff7f0e', label='Synthetic', linewidth=2)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(q_labels)
    axes[0].set_title('Minutiae Survival Rate (%)', fontweight='bold')
    axes[0].set_ylabel('Survival %')
    axes[0].set_ylim(80, 102)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # 2. Position Noise
    r_pos = [0.0] + [real_quality_stats[q]['pos_err'] for q in [7, 5, 1, 9]]
    s_pos = [0.0] + [synth_quality_stats[q]['pos_err'] for q in [7, 5, 1, 9]]
    axes[1].plot(range(5), r_pos, 'o-', color='#2ca02c', label='Real Data', linewidth=2)
    axes[1].plot(range(5), s_pos, 's--', color='#d62728', label='Synthetic', linewidth=2)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(q_labels)
    axes[1].set_title('Mean Localization Error (px)', fontweight='bold')
    axes[1].set_ylabel('Error (pixels)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    # 3. False Minutiae Count
    r_false = [0.0] + [real_quality_stats[q]['false'] for q in [7, 5, 1, 9]]
    s_false = [0.0] + [synth_quality_stats[q]['false'] for q in [7, 5, 1, 9]]
    axes[2].plot(range(5), r_false, 'o-', color='#9467bd', label='Real Data', linewidth=2)
    axes[2].plot(range(5), s_false, 's--', color='#8c564b', label='Synthetic', linewidth=2)
    axes[2].set_xticks(range(5))
    axes[2].set_xticklabels(q_labels)
    axes[2].set_title('Spurious Minutiae Count', fontweight='bold')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
