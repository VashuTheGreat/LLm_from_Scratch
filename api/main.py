import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipelines.ModelPredictionPipeline import ModelPredictionPipeline
from src.entity.config_entity import ModelPredictionConfig
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LLM Chat", page_icon="🤖")
st.title("LLM Chat Interface")

with st.sidebar:
    st.header("Configuration")
    model_path_input = st.text_input("Model Path", value=os.getenv("MODEL_PATH", ""))
    top_k = st.slider("Top K", min_value=1, max_value=100, value=40)
    max_new_tokens = st.slider("Max New Tokens", min_value=10, max_value=500, value=200)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.8)

@st.cache_resource
def load_model(path):
    if not path or not os.path.exists(path):
        return None
    try:
        config = ModelPredictionConfig(model_path=path)
        return ModelPredictionPipeline(config)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

pipeline = load_model(model_path_input)

if pipeline is None:
    st.warning("Please provide a valid model path in the sidebar to start.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        context = ""
        for m in st.session_state.messages:
            if m["role"] == "user":
                context += f"User: {m['content']}\n"
            else:
                context += f"Assistant: {m['content']}\n"
        context += "Assistant:"

        full_response = ""
        try:
            for token in pipeline.predict_yeild(text=context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error during generation: {e}")
        
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})