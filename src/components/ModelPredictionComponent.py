import logging
from src.entity.config_entity import ModelPredictionConfig
from src.entity.estimator import MyModel
from src.constants import MAX_NEW_TOKEN, TEMPERATURE, TOP_K
from exception import MyException

class ModelPredictionComponent:
    def __init__(self, model_prediction_config: ModelPredictionConfig):
        logging.info("Initializing ModelPredictionComponent.")
        self.model_prediction_config = model_prediction_config
        self.my_model = MyModel()
        logging.info(f"Loading model from {self.model_prediction_config.model_path}")
        self.my_model.load_model(self.model_prediction_config.model_path)

    def predict(self, text: str,max_new_tokens:int=MAX_NEW_TOKEN) -> str:
        try:
            logging.info(f"Predicting for text: {text}")
            prediction = self.my_model.predict(text=text,max_new_tokens=max_new_tokens)
            logging.info("Prediction successful.")
            return prediction
        except Exception as e:
            logging.error(f"Error in ModelPredictionComponent: {str(e)}")
            raise MyException(f"Error in ModelPredictionComponent: {str(e)}")

    def predict_yeild(self, text: str,max_new_tokens:int=MAX_NEW_TOKEN,temperature:float=TEMPERATURE,top_k:int=TOP_K):
        try:
            logging.info(f"Predicting for text (yield): {text}")
            prediction_generator = self.my_model.predict_yeild(text=text,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k)
            logging.info("Prediction generator created.")
            return prediction_generator
        except Exception as e:
            logging.error(f"Error in ModelPredictionComponent: {str(e)}")
            raise MyException(f"Error in ModelPredictionComponent: {str(e)}")