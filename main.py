import os
import sys
from logger import *
import logging
from dotenv import load_dotenv

load_dotenv()

from src.pipelines.ModelPredictionPipeline import ModelPredictionPipeline
from src.entity.config_entity import ModelPredictionConfig

if not sys.argv or len(sys.argv) <= 1:
    raise ValueError("No model path found")

model_path = sys.argv[1]
model_prediction_config = ModelPredictionConfig(model_path=model_path)
model_prediction_pipeline = ModelPredictionPipeline(model_prediction_config)

is_terminal = os.getenv("TERMINAL")

top_k = 40
max_new_tokens = 200
temperature = 0.8

messages = []

USER_COLOR = "\033[94m"
AI_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"

if is_terminal:
    while True:
        try:
            user_m = input(f"\n{USER_COLOR}User: {RESET_COLOR}")
            if user_m.strip().lower() in ['exit', 'quit']:
                break
            
            messages.append(f"User: {user_m}")
            content = "\n".join(messages) + "\nAssistant:"
            
            y = model_prediction_pipeline.predict_yeild(text=content, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            
            print(f"{AI_COLOR}Assistant: ", end="", flush=True)
            ai_m = "Assistant:"
            
            for to in y:
                ai_m += to
                print(to, end="", flush=True)
            print(RESET_COLOR)
            
            messages.append(ai_m)
            
        except KeyboardInterrupt:
            print(f"{RESET_COLOR}\nExiting...")
            break
else:
    from api.main import app