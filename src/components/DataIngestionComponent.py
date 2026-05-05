from datasets import load_dataset
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from exception import MyException
import os
import logging

class DataIngestionComponent:
    def __init__(self, data_Ingestion_config:DataIngestionConfig):
        logging.info("Initializing DataIngestionComponent.")
        self.data_Ingestion_config = data_Ingestion_config
        logging.info(f"DataIngestionComponent config set: {self.data_Ingestion_config}")

    def ingest(self)->DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process.")
            raw_data_path = self.data_Ingestion_config.raw_data_file_path
            logging.info(f"Reading raw data from: {raw_data_path}")
            with open(raw_data_path, 'r') as file:
                data = file.read()
            logging.info("Successfully read raw data file.")
            
            ingested_file_path = self.data_Ingestion_config.ingested_data_file_path
            dir_path = os.path.dirname(ingested_file_path)
            logging.info(f"Ensuring directory exists: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Writing ingested data to: {ingested_file_path}")
            with open(ingested_file_path, 'w') as file:
                file.write(data)
            logging.info("Successfully wrote ingested data.")
            
            data_Ingestion_artifact = DataIngestionArtifact(ingested_data_file_path=ingested_file_path)
            logging.info(f"Data ingestion artifact created: {data_Ingestion_artifact}")
            return data_Ingestion_artifact
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise MyException(f"Error during data Ingestion: {str(e)}") 