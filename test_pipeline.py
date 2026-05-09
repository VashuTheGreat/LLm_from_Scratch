import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.pipelines.ModelPredictionPipeline import ModelPredictionPipeline
from src.entity.config_entity import ModelPredictionConfig

config = ModelPredictionConfig(model_path="artifact/model.pth")
pipeline = ModelPredictionPipeline(config)

context = "User: tell me a joke\nAssistant:"
print("Yielding:")
try:
    for token in pipeline.predict_yeild(text=context, max_new_tokens=50, temperature=0.8, top_k=40):
        print(f"YIELDED: {repr(token)}")
except Exception as e:
    print("ERROR:", e)
