"""
Models package for synthetic fingerprint minutiae generation:
- Model A: Identity Generator
- Model B: Impression (Angle) & Quality Degrader
- Minutia Correspondence & Alignment
"""
from src.models.correspondence import MinutiaMatcher, CorrespondenceResult
from src.models.spatial_model import (
    BaseSpatialModel,
    GaussianMixtureSpatialModel,
    AdaptiveKDESpatialModel,
    InhomogeneousPoissonInhibitionModel,
)
from src.models.orientation_model import OrientationFieldModel
from src.models.type_model import MinutiaTypeModel
from src.models.model_a import ModelA, SyntheticIdentity
from src.models.angle_model import AngleModel, TransformationParameters
from src.models.quality_model import QualityModel, QualityParameters
