"""
Dataset manifest manager and resumable multi-shard coordinator.
Logically unifies multiple HDF5 shards into a single scalable dataset and coordinates resumability.
"""
from datetime import datetime, timezone
import json
import os
from dataclasses import dataclass, asdict

@dataclass
class ShardInfo:
    shard_id: int
    filename: str
    num_observations: int
    num_identities: int
    identity_start: int
    identity_end: int
    status: str # "completed", "in_progress"

class DatasetManifest:
    """
    Manages manifest.json across HDF5 shards.
    """
    def __init__(self, output_dir: str, master_seed: int = 42, model_version: str = "1.0.0"):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.master_seed = master_seed
        self.model_version = model_version
        self.shards: list[ShardInfo] = []
        self._load_or_init()

    def _load_or_init(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                data = json.load(f)
            self.master_seed = data.get('master_seed', self.master_seed)
            self.model_version = data.get('model_version', self.model_version)
            self.shards = [ShardInfo(**s) for s in data.get('shards', [])]
        else:
            self.save()

    @property
    def total_observations(self) -> int:
        return sum(s.num_observations for s in self.shards if s.status == "completed")

    @property
    def total_identities(self) -> int:
        return sum(s.num_identities for s in self.shards if s.status == "completed")

    def save(self):
        data = {
            'master_seed': self.master_seed,
            'model_version': self.model_version,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'total_observations': sum(s.num_observations for s in self.shards if s.status == "completed"),
            'total_identities': sum(s.num_identities for s in self.shards if s.status == "completed"),
            'num_shards': len(self.shards),
            'shards': [asdict(s) for s in self.shards]
        }
        temp_path = self.manifest_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.manifest_path)

    def record_shard(self, shard_info: ShardInfo):
        """
        Records or updates a shard in the manifest.
        """
        # Replace if exists, else append
        for i, s in enumerate(self.shards):
            if s.shard_id == shard_info.shard_id:
                self.shards[i] = shard_info
                self.save()
                return
        self.shards.append(shard_info)
        self.save()

    def get_resume_state(self, requested_total_identities: int) -> tuple[int, int]:
        """
        Calculates resumable state.
        Returns (next_identity_to_generate, next_shard_id).
        """
        completed = [s for s in self.shards if s.status == "completed"]
        if not completed:
            return 1, 0
            
        # Shards are sorted by shard_id
        completed_sorted = sorted(completed, key=lambda s: s.shard_id)
        last_shard = completed_sorted[-1]
        next_identity = last_shard.identity_end + 1
        next_shard_id = last_shard.shard_id + 1
        
        return next_identity, next_shard_id

    def locate_identity(self, identity: int) -> tuple[str, int] | None:
        """
        Finds the shard file and starting observation offset for a given identity.
        """
        for s in self.shards:
            if s.status == "completed" and s.identity_start <= identity <= s.identity_end:
                shard_path = os.path.join(self.output_dir, s.filename)
                obs_offset = (identity - s.identity_start) * 15
                return shard_path, obs_offset
        return None
