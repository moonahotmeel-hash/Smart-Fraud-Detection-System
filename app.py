import gradio as gr
import pickle
from tensorflow.keras.models import load_model

# Load Model
model = load_model("models/fraud_detection_model.keras")

# Load Scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load Encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def system_status():
    return "✅ Fraud Detection System Loaded Successfully"

demo = gr.Interface(
    fn=system_status,
    inputs=[],
    outputs="text",
    title="Smart Fraud Detection System"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
