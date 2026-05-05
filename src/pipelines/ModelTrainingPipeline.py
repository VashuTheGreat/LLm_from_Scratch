import logging
from src.components.ModelTrainingComponent import ModelTrainingComponent
from src.entity.artifact_entity import ModelTrainingArtifact, DataTransformationArtifact
from exception import MyException
from src.entity.config_entity import ModelTrainingConfig

class ModelTrainingPipeline:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_training_config: ModelTrainingConfig):
        logging.info("Initializing ModelTrainingPipeline.")
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_config = model_training_config
        self.model_training_component = ModelTrainingComponent(self.data_transformation_artifact, self.model_training_config)

    def initiate(self) -> ModelTrainingArtifact:
        try:
            logging.info("Starting ModelTrainingPipeline initiate method.")
            model_training_artifact = self.model_training_component.train_model()
            logging.info(f"ModelTrainingComponent finished successfully. Artifact: {model_training_artifact}")
            return model_training_artifact
        except Exception as e:
            logging.error(f"Error in ModelTrainingPipeline: {str(e)}")
            raise MyException(f"Error in Model Training Pipeline: {str(e)}")
