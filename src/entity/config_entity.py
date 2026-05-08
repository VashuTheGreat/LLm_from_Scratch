from dataclasses import dataclass
import os
from src.constants import *


@dataclass
class GPTConfig:
    vocab_size:int= VOCAB_SIZE   # Vocabulary size
    context_length: int= CONTEXT_LENGTH # Shortened context length (orig: 1024)
    emb_dim: int= EMB_DIM        # Embedding dimension
    n_heads: int= N_HEADS         # Number of attention heads
    n_layers: int= N_LAYERS        # Number of layers
    drop_rate: float= DROP_RATE     # Dropout rate
    qkv_bias: bool= QKV_BIAS   # Whether to include bias in QKV projections

    

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
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_epochs: int = NUM_EPOCHS
    eval_freq: int = EVAL_FREQ
    eval_iter: int = EVAL_ITER
    start_context: str = START_CONTEXT
    batch_size: int = BATCH_SIZE
    max_length: int = MAX_LENGTH
    stride: int = STRIDE
    num_workers: int = NUM_WORKERS
    load_pretrained: bool = LOAD_PRETRAINED   # Whether to load pretrained weights

@dataclass
class ModelPredictionConfig:
    model_path: str