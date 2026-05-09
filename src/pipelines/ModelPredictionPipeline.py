import logging
from src.components.ModelPredictionComponent import ModelPredictionComponent
from exception import MyException
from src.entity.config_entity import ModelPredictionConfig
from src.constants import MAX_NEW_TOKEN, TEMPERATURE, TOP_K

class ModelPredictionPipeline:
    def __init__(self, model_prediction_config: ModelPredictionConfig):
        logging.info("Initializing ModelPredictionPipeline.")
        self.model_prediction_config = model_prediction_config
        self.model_prediction_component = ModelPredictionComponent(self.model_prediction_config)

    def predict(self, text: str,max_new_tokens:int=MAX_NEW_TOKEN) -> str:
        try:
            logging.info("Starting ModelPredictionPipeline predict method.")
            prediction = self.model_prediction_component.predict(text,max_new_tokens)
            logging.info("ModelPredictionComponent finished successfully.")
            return prediction
        except Exception as e:
            logging.error(f"Error in ModelPredictionPipeline: {str(e)}")
            raise MyException(f"Error in Model Prediction Pipeline: {str(e)}")

    def predict_yeild(self, text: str,max_new_tokens:int=MAX_NEW_TOKEN,temperature:float=TEMPERATURE,top_k:int=TOP_K):
        try:
            logging.info("Starting ModelPredictionPipeline predict_yeild method.")
            prediction_generator = self.model_prediction_component.predict_yeild(text,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k)
            logging.info("ModelPredictionComponent finished successfully.")
            return prediction_generator
        except Exception as e:
            logging.error(f"Error in ModelPredictionPipeline: {str(e)}")
            raise MyException(f"Error in Model Prediction Pipeline: {str(e)}")

   
