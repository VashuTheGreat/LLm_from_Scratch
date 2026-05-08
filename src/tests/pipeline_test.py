import os
import sys
sys.path.append(os.getcwd())
from logger import *
import logging

from src.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelTrainingConfig,ModelPredictionConfig
from src.pipelines.DataIngestionPipeline import DataIngestionPipeline
from src.pipelines.DataTransformationPipeline import DataTransformationPipeline
from src.pipelines.ModelTrainingPipeline import ModelTrainingPipeline
from src.pipelines.ModelPredictionPipeline import ModelPredictionPipeline

def main():
    print("received args",sys.argv)
    if not sys.argv or len(sys.argv)<=1:
        raise ValueError("No dataset path found")
    logging.info("Pipeline test started.")
    # data_ingestion_config=DataIngestionConfig(raw_data_file_path="data/the-verdict.txt")
    data_ingestion_config=DataIngestionConfig(raw_data_file_path=sys.argv[1])
    data_ingestion_pipeline=DataIngestionPipeline(data_ingestion_config)
    data_ingestion_artifact=data_ingestion_pipeline.initiate()
    logging.info(f"Data Ingestion completed. Artifact: {data_ingestion_artifact}")


    data_Transformation_config=DataTransformationConfig()
    data_Transformation_pipeline=DataTransformationPipeline(data_ingestion_artifact, data_Transformation_config)
    data_Transformation_artifact=data_Transformation_pipeline.initiate()
    logging.info(f"Data Transformation completed. Artifact: {data_Transformation_artifact}")
    
    model_training_config=ModelTrainingConfig()
    model_training_pipeline=ModelTrainingPipeline(data_Transformation_artifact, model_training_config)
    model_training_artifact=model_training_pipeline.initiate()
    logging.info(f"Model Training completed. Artifact: {model_training_artifact}")

    model_prediction_config=ModelPredictionConfig(model_path=model_training_artifact.trained_model_file_path)
    model_prediction_pipeline=ModelPredictionPipeline(model_prediction_config)
    prediction=model_prediction_pipeline.predict("Every effort moves you")
    logging.info(f"Model Prediction completed. Prediction: {prediction}")

    logging.info("Pipeline test completed successfully.")

if __name__ == "__main__":
    main()