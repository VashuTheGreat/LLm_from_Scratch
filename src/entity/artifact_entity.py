

from dataclasses import dataclass
from src.constants import *
import os
@dataclass
class DataIngestionArtifact:
    ingested_data_file_path:str

@dataclass
class DataTransformationArtifact:
    train_tokens_path: str
    test_tokens_path: str

@dataclass
class ModelTrainingArtifact:
    trained_model_file_path: str

@dataclass
class ModelPredictionArtifact:
    prediction: str