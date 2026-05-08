import torch
import yaml
import os
try:
    with open(os.path.join("config","model.yml"), "r") as file:
        model_config = yaml.safe_load(file)
        print(model_config)
except Exception as e:
    print(e)        

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



TRAIN_TEST_SPLIT_RATIO= model_config['MODEL_CONFIG']['TRAIN_TEST_SPLIT_RATIO']
VOCAB_SIZE= model_config['MODEL_CONFIG']['VOCAB_SIZE']        # Vocabulary size
CONTEXT_LENGTH= model_config['MODEL_CONFIG']['CONTEXT_LENGTH']      # Shortened context length (orig: 1024)
EMB_DIM= model_config['MODEL_CONFIG']['EMB_DIM']             # Embedding dimension
N_HEADS= model_config['MODEL_CONFIG']['N_HEADS']              # Number of attention heads
N_LAYERS= model_config['MODEL_CONFIG']['N_LAYERS']             # Number of layers
DROP_RATE= model_config['MODEL_CONFIG']['DROP_RATE']          # Dropout rate
QKV_BIAS= model_config['MODEL_CONFIG']['QKV_BIAS'] 
LEARNING_RATE=float(model_config['MODEL_CONFIG']['LEARNING_RATE'])
WEIGHT_DECAY=float(model_config['MODEL_CONFIG']['WEIGHT_DECAY'])
NUM_EPOCHS=model_config['MODEL_CONFIG']['NUM_EPOCHS']
EVAL_FREQ=model_config['MODEL_CONFIG']['EVAL_FREQ']
EVAL_ITER=model_config['MODEL_CONFIG']['EVAL_ITER']
START_CONTEXT=model_config['MODEL_CONFIG']['START_CONTEXT']
BATCH_SIZE=model_config['MODEL_CONFIG']['BATCH_SIZE']
MAX_LENGTH=model_config['MODEL_CONFIG']['MAX_LENGTH']
STRIDE=model_config['MODEL_CONFIG']['STRIDE']
NUM_WORKERS=model_config['MODEL_CONFIG']['NUM_WORKERS']


# encoder name
TOKENIZER_NAME=model_config['MODEL_CONFIG']['TOKENIZER_NAME']
ENCODER_NAME=model_config['MODEL_CONFIG']['TOKENIZER_NAME']

# MODEL

LOAD_PRETRAINED=model_config['MODEL_CONFIG']['LOAD_PRETRAINED_WEIGHTS']
TEMPERATURE=0.8
TOP_K=200
MAX_NEW_TOKEN=6
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
