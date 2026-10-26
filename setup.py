from setuptools import setup, find_packages

setup(
    name="duplicate_detection",
    version="1.0.0",
    description="Large-Scale Synthetic Fingerprint Generation & LSH Bloom Filter Duplicate Detection",
    author="HPC Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ]
    },
)
