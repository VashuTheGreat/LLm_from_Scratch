import torch

ARTIFACT_DIR="artifact"

DATA_INGESTION_DIR="data_ingestion"
DATA_TRANSFORMATION_DIR="data_transformation"
MODEL_TRAINING_DIR="model_training"
MODEL_PREDICTION_DIR="model_prediction"
DATA_LOADER_FILE_NAME="data_loader.pkl"


ALLOWED_SPECIAL_TOKEN="<|endoftext|>"

DATA_TRAIN_FILE_NAME="train.npy"
DATA_TEST_FILE_NAME="test.npy"
TRAINED_MODEL_FILE_NAME="model.pth"

# encoder name
TOKENIZER_NAME="gpt2"
ENCODER_NAME="gpt2"
TRAIN_TEST_SPLIT_RATIO=0.1

# MODEL
TEMPERATURE=0.8
TOP_K=200
MAX_NEW_TOKEN=6
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
