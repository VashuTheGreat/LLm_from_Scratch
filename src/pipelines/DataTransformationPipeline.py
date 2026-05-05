import logging
from src.components.DataTransformationComponent import DataTransformationComponent
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from exception import MyException
from src.entity.config_entity import DataTransformationConfig

class DataTransformationPipeline:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_Transformation_config: DataTransformationConfig):
        logging.info("Initializing DataTransformationPipeline with config.")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_Transformation_config = data_Transformation_config
        logging.info(f"Config set: {self.data_Transformation_config}")
        self.data_Transformation_component = DataTransformationComponent(self.data_ingestion_artifact, self.data_Transformation_config)
        logging.info("DataTransformationComponent instantiated.")

    def initiate(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting DataTransformationPipeline initiate method.")
            data_Transformation_artifact = self.data_Transformation_component.transform()
            logging.info(f"DataTransformationComponent transform method completed successfully. Artifact: {data_Transformation_artifact}")
            return data_Transformation_artifact
        except Exception as e:
            logging.error(f"Error in DataTransformationPipeline: {str(e)}")
            raise MyException(f"Error in Data Transformation Pipeline: {str(e)}")