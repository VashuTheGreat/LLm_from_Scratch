import logging
from src.components.ModelPredictionComponent import ModelPredictionComponent
from exception import MyException
from src.entity.config_entity import ModelPredictionConfig

class ModelPredictionPipeline:
    def __init__(self, model_prediction_config: ModelPredictionConfig):
        logging.info("Initializing ModelPredictionPipeline.")
        self.model_prediction_config = model_prediction_config
        self.model_prediction_component = ModelPredictionComponent(self.model_prediction_config)

    def predict(self, text: str) -> str:
        try:
            logging.info("Starting ModelPredictionPipeline predict method.")
            prediction = self.model_prediction_component.predict(text)
            logging.info("ModelPredictionComponent finished successfully.")
            return prediction
        except Exception as e:
            logging.error(f"Error in ModelPredictionPipeline: {str(e)}")
            raise MyException(f"Error in Model Prediction Pipeline: {str(e)}")

   
