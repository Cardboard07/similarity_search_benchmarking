"""
Interoperability exporter.
Exports synthetic fingerprint observations from HDF5 shards to individual JSON files
preserving the original dataset naming convention: q<quality>_<identity>_<orientation>.json
and exact JSON schema.
"""
import json
import os
from src.storage.manifest import DatasetManifest
from src.storage.hdf5_reader import HDF5ShardReader

class JSONExporter:
    """
    Exports synthetic fingerprint observations from HDF5 shards into individual JSON files.
    """
    def __init__(self, manifest_path: str):
        self.output_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.manifest = DatasetManifest(self.output_dir)

    def export_identity(
        self,
        identity: int,
        export_dir: str
    ) -> list[str]:
        """
        Exports all 15 observations of a synthetic identity into JSON files in export_dir.
        Returns list of created file paths.
        """
        located = self.manifest.locate_identity(identity)
        if located is None:
            raise ValueError(f"Identity {identity} not found in manifest: {self.manifest.manifest_path}")
            
        shard_path, obs_offset = located
        os.makedirs(export_dir, exist_ok=True)
        created_files = []

        with HDF5ShardReader(shard_path) as reader:
            for obs_i in range(15):
                sample = reader.get_sample(obs_offset + obs_i)
                q = sample['quality']
                ident = sample['identity']
                ori = sample['orientation']
                c = sample['num_minutiae']
                
                filename = f"q{q}_{ident}_{ori}.json"
                file_path = os.path.join(export_dir, filename)
                
                minutiae_list = []
                for m_idx in range(c):
                    minutiae_list.append({
                        'x': int(round(sample['pts'][m_idx, 0])),
                        'y': int(round(sample['pts'][m_idx, 1])),
                        'angle': float(sample['angles'][m_idx]),
                        'type': int(sample['types'][m_idx]),
                        'quality': int(sample['qualities'][m_idx])
                    })
                    
                data = {
                    'file_name': filename,
                    'quality_folder': q,
                    'identity': str(ident),
                    'impression': str(ori),
                    'image_size': {
                        'x': 416,
                        'y': 560
                    },
                    'num_minutiae': c,
                    'minutiae': minutiae_list
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                created_files.append(file_path)

        return created_files

    def export_identity_range(
        self,
        start_identity: int,
        end_identity: int,
        export_dir: str
    ) -> list[str]:
        """
        Exports a range of synthetic identities [start_identity, end_identity] (inclusive).
        """
        all_created = []
        for ident in range(start_identity, end_identity + 1):
            created = self.export_identity(ident, export_dir)
            all_created.extend(created)
        return all_created
