"""
Streaming generation pipeline for millions of synthetic fingerprints.
Generates structured batches in memory and yields arrays ready for chunked HDF5 writing.
"""
from dataclasses import dataclass
import hashlib
from typing import Iterator
import numpy as np

from src.models.model_a import ModelA
from src.models.angle_model import AngleModel
from src.models.quality_model import QualityModel
from src.generation.identity import generate_pristine_identity
from src.generation.angles import generate_impressions
from src.generation.quality import generate_quality_observations

@dataclass
class GenerationBatch:
    minutiae: np.ndarray       # [N_obs, MAX_M, 5] float32
    num_minutiae: np.ndarray   # [N_obs] int32
    identities: np.ndarray     # [N_obs] int32
    orientations: np.ndarray   # [N_obs] int32
    qualities: np.ndarray      # [N_obs] int32
    seeds: np.ndarray          # [N_obs] int64
    num_identities: int
    num_observations: int

class SyntheticGenerationPipeline:
    """
    Coordinates Model A, B_angle, and B_quality into a scalable streaming batch generator.
    """
    def __init__(
        self,
        model_a: ModelA,
        angle_model: AngleModel,
        quality_model: QualityModel,
        max_minutiae: int = 128
    ):
        self.model_a = model_a
        self.angle_model = angle_model
        self.quality_model = quality_model
        self.max_minutiae = max_minutiae

    @staticmethod
    def derive_seed(master_seed: int, identity_idx: int) -> int:
        """
        Deterministic seed derivation using cryptographic hashing.
        Guarantees independent, reproducible streams across identities and parallel workers.
        """
        key = f"{master_seed}_{identity_idx}".encode('utf-8')
        h = hashlib.sha256(key).digest()
        # Take 31-bit integer for standard RNG compatibility
        return int(np.frombuffer(h[:4], dtype=np.uint32)[0] & 0x7FFFFFFF)

    def generate_batch(
        self,
        start_identity: int,
        num_identities_in_batch: int,
        master_seed: int
    ) -> GenerationBatch:
        """
        Generates a single batch of identities and all their 15 observations.
        """
        n_obs = num_identities_in_batch * 15
        
        minutiae_arr = np.zeros((n_obs, self.max_minutiae, 5), dtype=np.float32)
        num_minutiae_arr = np.zeros(n_obs, dtype=np.int32)
        identities_arr = np.zeros(n_obs, dtype=np.int32)
        orientations_arr = np.zeros(n_obs, dtype=np.int32)
        qualities_arr = np.zeros(n_obs, dtype=np.int32)
        seeds_arr = np.zeros(n_obs, dtype=np.int64)

        obs_idx = 0
        for i in range(num_identities_in_batch):
            ident = start_identity + i
            id_seed = self.derive_seed(master_seed, ident)
            
            # Model A: Pristine identity
            pristine = generate_pristine_identity(self.model_a, id_seed)
            
            # Model B_angle: 3 impressions
            impressions = generate_impressions(pristine, self.angle_model, id_seed)
            
            # Model B_quality: 5 quality levels for each impression
            for imp in impressions:
                observations = generate_quality_observations(imp, self.quality_model, id_seed + imp['orientation'] * 73)
                for obs in observations:
                    c = min(obs['num_minutiae'], self.max_minutiae)
                    # Pack features: [x, y, angle, type, quality]
                    if c > 0:
                        minutiae_arr[obs_idx, :c, 0] = obs['pts'][:c, 0]
                        minutiae_arr[obs_idx, :c, 1] = obs['pts'][:c, 1]
                        minutiae_arr[obs_idx, :c, 2] = obs['angles'][:c]
                        minutiae_arr[obs_idx, :c, 3] = obs['types'][:c]
                        minutiae_arr[obs_idx, :c, 4] = obs['qualities'][:c]
                    
                    num_minutiae_arr[obs_idx] = c
                    identities_arr[obs_idx] = ident
                    orientations_arr[obs_idx] = obs['orientation']
                    qualities_arr[obs_idx] = obs['quality']
                    seeds_arr[obs_idx] = id_seed
                    obs_idx += 1

        return GenerationBatch(
            minutiae=minutiae_arr,
            num_minutiae=num_minutiae_arr,
            identities=identities_arr,
            orientations=orientations_arr,
            qualities=qualities_arr,
            seeds=seeds_arr,
            num_identities=num_identities_in_batch,
            num_observations=n_obs
        )

    def generate_streaming(
        self,
        total_identities: int,
        batch_size: int,
        master_seed: int,
        start_identity: int = 1
    ) -> Iterator[GenerationBatch]:
        """
        Streaming generator that yields batches sequentially without storing all in RAM.
        """
        generated = 0
        while generated < total_identities:
            current_batch_size = min(batch_size, total_identities - generated)
            batch = self.generate_batch(
                start_identity=start_identity + generated,
                num_identities_in_batch=current_batch_size,
                master_seed=master_seed
            )
            yield batch
            generated += current_batch_size
