import logging
from src.entity.config_entity import ModelPredictionConfig
from src.entity.estimator import MyModel
from exception import MyException

class ModelPredictionComponent:
    def __init__(self, model_prediction_config: ModelPredictionConfig):
        logging.info("Initializing ModelPredictionComponent.")
        self.model_prediction_config = model_prediction_config
        self.my_model = MyModel()
        logging.info(f"Loading model from {self.model_prediction_config.model_path}")
        self.my_model.load_model(self.model_prediction_config.model_path)

    def predict(self, text: str) -> str:
        try:
            logging.info(f"Predicting for text: {text}")
            prediction = self.my_model.predict(text=text)
            logging.info("Prediction successful.")
            return prediction
        except Exception as e:
            logging.error(f"Error in ModelPredictionComponent: {str(e)}")
            raise MyException(f"Error in ModelPredictionComponent: {str(e)}")