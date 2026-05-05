import os
import sys
sys.path.append(os.getcwd())
from logger import *
import logging
from src.entity.config_entity import ModelPredictionConfig
from src.entity.artifact_entity import ModelTrainingArtifact
from src.pipelines.ModelPredictionPipeline import ModelPredictionPipeline

def main():
    
    model_path:str="/home/vashuthegreat/Projects/LLM_from_Scratch/artifact/20260505-115753/model_training/model.pth"
    model_prediction_config=ModelPredictionConfig(model_path=model_path)
    model_prediction_pipeline=ModelPredictionPipeline(model_prediction_config)
    prediction=model_prediction_pipeline.predict("Every effort moves you")
    logging.info(f"Model Prediction completed. Prediction: {prediction}")

    logging.info("Pipeline test completed successfully.")

if __name__ == "__main__":
    main()