import os
import pickle
import gradio as gr
from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL AND FILES
# ==========================

try:
    model = load_model("models/fraud_detection_model.keras")

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    STATUS = "✅ Smart Fraud Detection System Loaded Successfully"

except Exception as e:
    STATUS = f"❌ Error Loading System:\n{str(e)}"


# ==========================
# SYSTEM STATUS FUNCTION
# ==========================

def system_status():
    return STATUS


# ==========================
# GRADIO INTERFACE
# ==========================

demo = gr.Interface(
    fn=system_status,
    inputs=[],
    outputs="text",
    title="Smart Fraud Detection System",
    description="CNN-RNN-LSTM Hybrid Model for Financial Fraud Detection"
)


# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
