"""
Dataset loader and calibration engine.
Extracts empirical distributions and ground-truth parameters across all 1,000 identities
and their 15 observations in the real fingerprint dataset.
"""
import glob
import json
import os
import re
from tqdm import tqdm
import numpy as np

from src.statistics.spatial_stats import (
    nearest_neighbor_distances,
    pairwise_distances,
    local_densities,
    boundary_distances
)
from src.statistics.circular_stats import (
    circular_mean_deg,
    circular_std_deg,
    circular_abs_distance_deg,
    fit_von_mises
)

class DatasetCalibrator:
    """
    Scans the source JSON files and computes comprehensive target calibration statistics.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_map: dict[str, str] = {} # filename -> full_path
        self._index_files()

    def _index_files(self):
        # Look in train and test subfolders
        patterns = [
            os.path.join(self.data_dir, "*.json"),
            os.path.join(self.data_dir, "*", "*.json")
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
            
        for f in files:
            self.file_map[os.path.basename(f)] = f

    def load_fingerprint_file(self, filename: str) -> dict | None:
        path = self.file_map.get(filename)
        if not path or not os.path.exists(path):
            return None
        with open(path, 'r') as fp:
            d = json.load(fp)
            
        pts = np.array([[m['x'], m['y']] for m in d['minutiae']], dtype=float)
        angles = np.array([m['angle'] for m in d['minutiae']], dtype=float)
        types = np.array([m['type'] for m in d['minutiae']], dtype=int)
        qualities = np.array([m['quality'] for m in d['minutiae']], dtype=int)
        
        return {
            'identity': int(d.get('identity', 0)),
            'orientation': int(d.get('impression', 1)),
            'quality': int(d.get('quality_folder', 3)),
            'pts': pts,
            'angles': angles,
            'types': types,
            'qualities': qualities,
            'num_minutiae': len(pts)
        }

    def load_all_pristine(self) -> list[dict]:
        """
        Loads all 1000 pristine fingerprints (q3_<id>_1.json).
        """
        pristine = []
        for ident in range(1, 1001):
            fname = f"q3_{ident}_1.json"
            fp = self.load_fingerprint_file(fname)
            if fp is not None:
                pristine.append(fp)
        return pristine

    def run_full_calibration(self) -> dict:
        """
        Runs complete statistical calibration over pristine fingerprints, paired impressions, and quality levels.
        """
        print("Extracting pristine population statistics...")
        pristine_fps = self.load_all_pristine()
        
        counts = np.array([len(fp['pts']) for fp in pristine_fps])
        all_pts = []
        all_angles = []
        all_types = []
        all_nn_dists = []
        all_pairwise_dists = []
        all_local_densities = []
        all_boundary_dists = []

        for fp in pristine_fps:
            pts = fp['pts']
            if len(pts) > 0:
                all_pts.append(pts)
                all_angles.extend(fp['angles'])
                all_types.extend(fp['types'])
                
                nn = nearest_neighbor_distances(pts, k=1)
                all_nn_dists.extend(nn)
                pw = pairwise_distances(pts, max_pairs=50)
                all_pairwise_dists.extend(pw)
                dens = local_densities(pts, radius=30.0)
                all_local_densities.extend(dens)
                bd = boundary_distances(pts)
                all_boundary_dists.extend(bd)

        stacked_pts = np.vstack(all_pts)
        all_angles_arr = np.array(all_angles)
        all_types_arr = np.array(all_types)
        all_nn_arr = np.array(all_nn_dists)
        all_pw_arr = np.array(all_pairwise_dists)

        # Minutiae count statistics
        count_stats = {
            'mean': float(np.mean(counts)),
            'median': float(np.median(counts)),
            'variance': float(np.var(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
            'percentiles': {p: float(np.percentile(counts, p)) for p in [5, 10, 25, 50, 75, 90, 95]}
        }

        # Spatial statistics
        spatial_stats = {
            'nn_distance_mean': float(np.mean(all_nn_arr)),
            'nn_distance_median': float(np.median(all_nn_arr)),
            'nn_distance_std': float(np.std(all_nn_arr)),
            'nn_distance_min': float(np.min(all_nn_arr)),
            'nn_distance_5th': float(np.percentile(all_nn_arr, 5)),
            'pairwise_mean': float(np.mean(all_pw_arr)),
            'pairwise_std': float(np.std(all_pw_arr)),
            'density_mean': float(np.mean(all_local_densities)),
            'boundary_dist_mean': float(np.mean(all_boundary_dists)),
            'x_min': float(np.min(stacked_pts[:, 0])),
            'x_max': float(np.max(stacked_pts[:, 0])),
            'y_min': float(np.min(stacked_pts[:, 1])),
            'y_max': float(np.max(stacked_pts[:, 1])),
        }

        # Orientation statistics
        ang_mean = circular_mean_deg(all_angles_arr)
        ang_std = circular_std_deg(all_angles_arr)
        orientation_stats = {
            'circular_mean_deg': ang_mean,
            'circular_std_deg': ang_std
        }

        # Type proportions
        type_stats = {
            'type_1_ratio': float(np.mean(all_types_arr == 1)),
            'type_2_ratio': float(np.mean(all_types_arr == 2))
        }

        # Impression (Angle) correspondence analysis
        print("Calibrating impression / orientation variations across real pairs...")
        from src.models.correspondence import MinutiaMatcher, CorrespondenceResult
        matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)
        angle_results: list[CorrespondenceResult] = []
        
        for ident in range(1, 1001):
            fp1 = self.load_fingerprint_file(f"q3_{ident}_1.json")
            if not fp1:
                continue
            for ori in [2, 3]:
                fp_ori = self.load_fingerprint_file(f"q3_{ident}_{ori}.json")
                if fp_ori:
                    res = matcher.match(
                        pts_a=fp1['pts'],
                        angles_a=fp1['angles'],
                        types_a=fp1['types'],
                        pts_b=fp_ori['pts'],
                        angles_b=fp_ori['angles'],
                        types_b=fp_ori['types']
                    )
                    angle_results.append(res)

        tx_list = [r.estimated_tx for r in angle_results]
        ty_list = [r.estimated_ty for r in angle_results]
        rot_list = [r.estimated_rotation_deg for r in angle_results]
        scale_list = [r.estimated_scale for r in angle_results]
        ang_errs = [e for r in angle_results for e in r.angular_differences]

        angle_stats = {
            'tx_mean': float(np.mean(tx_list)),
            'tx_std': float(np.std(tx_list)),
            'ty_mean': float(np.mean(ty_list)),
            'ty_std': float(np.std(ty_list)),
            'rot_mean_deg': float(np.mean(rot_list)),
            'rot_std_deg': float(np.std(rot_list)),
            'scale_mean': float(np.mean(scale_list)),
            'scale_std': float(np.std(scale_list)),
            'angle_error_mean': float(np.mean(ang_errs)) if ang_errs else 0.0
        }

        # Quality degradation correspondence analysis
        print("Calibrating quality degradations across levels 7, 5, 1, 9...")
        quality_stats = {}
        for q in [7, 5, 1, 9]:
            q_surv = []
            q_false = []
            q_pos_err = []
            q_ang_err = []
            
            # Sample across identities
            for ident in range(1, 1001):
                fp3 = self.load_fingerprint_file(f"q3_{ident}_1.json")
                fq = self.load_fingerprint_file(f"q{q}_{ident}_1.json")
                if fp3 and fq:
                    res = matcher.match(
                        pts_a=fp3['pts'],
                        angles_a=fp3['angles'],
                        types_a=fp3['types'],
                        pts_b=fq['pts'],
                        angles_b=fq['angles'],
                        types_b=fq['types']
                    )
                    q_surv.append(res.survival_rate)
                    q_false.append(res.false_minutiae_count)
                    if len(res.spatial_distances) > 0:
                        q_pos_err.append(np.mean(res.spatial_distances))
                    if len(res.angular_differences) > 0:
                        q_ang_err.append(np.mean(res.angular_differences))

            quality_stats[q] = {
                'surv': float(np.mean(q_surv)),
                'false': float(np.mean(q_false)),
                'pos_err': float(np.mean(q_pos_err)) if q_pos_err else 0.0,
                'ang_err': float(np.mean(q_ang_err)) if q_ang_err else 0.0
            }

        return {
            'count_stats': count_stats,
            'spatial_stats': spatial_stats,
            'orientation_stats': orientation_stats,
            'type_stats': type_stats,
            'angle_stats': angle_stats,
            'quality_stats': quality_stats,
            'angle_correspondence_results': angle_results
        }
