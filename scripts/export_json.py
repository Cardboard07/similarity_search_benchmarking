"""
CLI script to export synthetic fingerprints from HDF5 shards into individual JSON files.
Preserves the exact source naming convention: q<quality>_<identity>_<orientation>.json
and canonical JSON schema.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.exporter import JSONExporter

def main():
    parser = argparse.ArgumentParser(description="Export synthetic fingerprints to individual JSON files.")
    parser.add_argument("--manifest", type=str, default="dataset/manifest.json", help="Path to dataset manifest.json")
    parser.add_argument("--output-dir", type=str, default="exported_json", help="Directory to save exported JSON files")
    parser.add_argument("--start-identity", type=int, default=1, help="First identity ID to export")
    parser.add_argument("--end-identity", type=int, default=5, help="Last identity ID to export (inclusive)")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    print("=" * 60)
    print(f"Exporting Synthetic Identities {args.start_identity} to {args.end_identity} to JSON")
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    exporter = JSONExporter(args.manifest)
    created_files = exporter.export_identity_range(
        start_identity=args.start_identity,
        end_identity=args.end_identity,
        export_dir=args.output_dir
    )

    print(f"\nSuccessfully exported {len(created_files)} JSON files (15 observations per identity).")
    print("Sample exported files:")
    for f in created_files[:10]:
        print(f"  {os.path.basename(f)}")
    if len(created_files) > 10:
        print(f"  ... and {len(created_files) - 10} more.")

if __name__ == '__main__':
    main()
