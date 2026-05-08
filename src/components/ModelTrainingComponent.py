import logging
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact
from src.entity.estimator import MyModel
from exception import MyException

class TokenDataset(Dataset):
    def __init__(self, token_path, max_length, stride):
        self.tokens = np.memmap(token_path, dtype=np.int32, mode='r')
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        if len(self.tokens) <= self.max_length:
            return 0
        return (len(self.tokens) - self.max_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length
        input_chunk = self.tokens[start:end].copy()
        target_chunk = self.tokens[start+1:end+1].copy()
        return torch.tensor(input_chunk, dtype=torch.long), torch.tensor(target_chunk, dtype=torch.long)

class ModelTrainingComponent:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_training_config: ModelTrainingConfig):
        logging.info("Initializing ModelTrainingComponent.")
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_config = model_training_config

    def train_model(self) -> ModelTrainingArtifact:
        try:
            logging.info("Starting model training.")
            
            # Load Data
            logging.info(f"Loading training data from {self.data_transformation_artifact.train_tokens_path}")
            train_dataset = TokenDataset(self.data_transformation_artifact.train_tokens_path, 
                                         max_length=self.model_training_config.max_length, 
                                         stride=self.model_training_config.stride)
                                         
            logging.info(f"Loading validation data from {self.data_transformation_artifact.test_tokens_path}")
            val_dataset = TokenDataset(self.data_transformation_artifact.test_tokens_path, 
                                       max_length=self.model_training_config.max_length, 
                                       stride=self.model_training_config.stride)
            
            logging.info(f"Train dataset length: {len(train_dataset)}")
            logging.info(f"Val dataset length: {len(val_dataset)}")
            
            train_loader = DataLoader(train_dataset, batch_size=self.model_training_config.batch_size, shuffle=True, num_workers=self.model_training_config.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=self.model_training_config.batch_size, shuffle=False, num_workers=self.model_training_config.num_workers)
            
            X = {
                "train": train_loader,
                "val": val_loader
            }
            
            my_model = MyModel(load_pretrained_weights=self.model_training_config.load_pretrained)
            
            logging.info("Fitting model.")
            my_model.fit(X, self.model_training_config)
            
            logging.info("Model training finished. Generating artifact.")
            
            artifact = ModelTrainingArtifact(
                trained_model_file_path=self.model_training_config.trained_model_file_path
            )
            return artifact
        except Exception as e:
            logging.error(f"Error in ModelTrainingComponent: {str(e)}")
            raise MyException(f"Error in ModelTrainingComponent: {str(e)}")