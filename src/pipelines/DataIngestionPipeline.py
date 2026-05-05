import logging
from src.components.DataIngestionComponent import DataIngestionComponent
from src.entity.artifact_entity import DataIngestionArtifact
from exception import MyException
from src.entity.config_entity import DataIngestionConfig

class DataIngestionPipeline:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        logging.info("Initializing DataIngestionPipeline with config.")
        self.data_ingestion_config=data_ingestion_config
        logging.info(f"Config set: {self.data_ingestion_config}")
        self.data_ingestion_component=DataIngestionComponent(self.data_ingestion_config)
        logging.info("DataIngestionComponent instantiated.")

    def initiate(self)->DataIngestionArtifact:
        try:
            logging.info("Starting DataIngestionPipeline initiate method.")
            data_ingestion_artifact=self.data_ingestion_component.ingest()
            logging.info(f"DataIngestionComponent ingest method completed successfully. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error in DataIngestionPipeline: {str(e)}")
            raise MyException(f"Error in Data Ingestion Pipeline: {str(e)}")