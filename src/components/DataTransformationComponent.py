from datasets import load_dataset
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact
from exception import MyException
import os
from src.constants import TOKENIZER_NAME
import numpy as np
import tiktoken
import logging

class DataTransformationComponent:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_Transformation_config:DataTransformationConfig):
        logging.info("Initializing DataTransformationComponent.")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_Transformation_config = data_Transformation_config
        logging.info(f"Loading encoder: {TOKENIZER_NAME}")
        self.encoder=tiktoken.get_encoding(TOKENIZER_NAME)
        logging.info("Encoder loaded successfully.")

    @staticmethod
    def train_test_split(txt, test_size=0.1):
        logging.info(f"Performing train test split with test_size={test_size}")
        total_length = len(txt)
        test_length = int(total_length * test_size)
        train_txt = txt[:-test_length]
        test_txt = txt[-test_length:]
        logging.info("Train test split completed.")
        return train_txt, test_txt

    def transform(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process.")
            # 1. streaming dataset load
            logging.info(f"Loading dataset from: {self.data_ingestion_artifact.ingested_data_file_path}")
            dataset = load_dataset(
                "text",
                data_files=self.data_ingestion_artifact.ingested_data_file_path,
                streaming=True
            )
            logging.info("Dataset loaded successfully.")

            # 2. prepare output paths
            train_path = self.data_Transformation_config.train_tokens_path
            test_path = self.data_Transformation_config.test_tokens_path
            logging.info(f"Train path set to: {train_path}")
            logging.info(f"Test path set to: {test_path}")

            logging.info(f"Ensuring directories exist for {train_path}")
            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            # 3. create memmap files (temporary large size assume)
            max_tokens = 10**9  # adjust if needed
            logging.info(f"Creating memmap files with max_tokens={max_tokens}")

            train_memmap = np.memmap(train_path, dtype=np.int32, mode='w+', shape=(max_tokens,))
            test_memmap = np.memmap(test_path, dtype=np.int32, mode='w+', shape=(max_tokens,))
            logging.info("Memmap files created.")

            train_idx, test_idx = 0, 0

            # 4. streaming loop
            logging.info("Starting streaming loop for dataset.")
            for sample in dataset["train"]:
                text = sample["text"]

                tokens = self.encoder.encode(text, allowed_special={"<|endoftext|>"})

                # split logic (simple ratio)
                split = int(len(tokens) * 0.9)

                train_tokens = tokens[:split]
                test_tokens = tokens[split:]

                # write to memmap
                train_memmap[train_idx:train_idx+len(train_tokens)] = train_tokens
                train_idx += len(train_tokens)

                test_memmap[test_idx:test_idx+len(test_tokens)] = test_tokens
                test_idx += len(test_tokens)
            logging.info("Streaming loop completed.")

            # 5. trim memmap (important)
            logging.info("Flushing memmap files.")
            train_memmap.flush()
            test_memmap.flush()

            # Close references
            del train_memmap
            del test_memmap

            # resize properly by truncating the file on disk
            logging.info(f"Truncating files. Train size: {train_idx}, Test size: {test_idx}")
            os.truncate(train_path, train_idx * 4)  # 4 bytes per int32
            os.truncate(test_path, test_idx * 4)
            logging.info("Memmap files resized and flushed.")

            # 6. return artifact
            artifact = DataTransformationArtifact(
                train_tokens_path=train_path,
                test_tokens_path=test_path
            )
            logging.info(f"DataTransformationArtifact created: {artifact}")
            return artifact

        except Exception as e:
            logging.error(f"Error during data Transformation: {str(e)}")
            raise MyException(f"Error during data Transformation: {str(e)}")