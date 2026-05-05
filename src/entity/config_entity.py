from dataclasses import dataclass
import os
from src.constants import *


@dataclass
class GPTConfig:
    vocab_size:int= 50257   # Vocabulary size
    context_length: int= 256 # Shortened context length (orig: 1024)
    emb_dim: int= 768        # Embedding dimension
    n_heads: int= 12         # Number of attention heads
    n_layers: int= 12        # Number of layers
    drop_rate: float= 0.1      # Dropout rate
    qkv_bias: bool= False   # Whether to include bias in QKV projections

    

# Data ingestion
import time
ARTIFACT_DIR=os.path.join(ARTIFACT_DIR,time.strftime("%Y%m%d-%H%M%S") )
@dataclass
class DataIngestionConfig:
    raw_data_file_path:str
    ingested_data_file_path:str=os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR, "ingested_data.txt")

@dataclass
class DataTransformationConfig:
    train_tokens_path: str=os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR, DATA_TRAIN_FILE_NAME)
    test_tokens_path: str=os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR, DATA_TEST_FILE_NAME)
    tokenizer_name: str = ENCODER_NAME
    test_size: float = TRAIN_TEST_SPLIT_RATIO

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINING_DIR, TRAINED_MODEL_FILE_NAME)
    dir_to_save_plots: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINING_DIR, "plots")
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    num_epochs: int = 1
    eval_freq: int = 5
    eval_iter: int = 5
    start_context: str = "Every effort moves you"
    batch_size: int = 2
    max_length: int = 256
    stride: int = 256
    num_workers: int = 0

@dataclass
class ModelPredictionConfig:
    model_path: str