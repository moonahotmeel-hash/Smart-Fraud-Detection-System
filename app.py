import gradio as gr
import pandas as pd
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================

model = load_model("models/fraud_detection_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# =====================================
# FEATURE ENGINEERING
# =====================================

def prepare_features(df):

    df["balanceDiffOrig"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

    df["balanceDiffDest"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["errorBalanceDest"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    return df

# =====================================
# SINGLE PREDICTION
# =====================================

def predict_fraud(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    isFlaggedFraud
):

    type_encoded = encoder.transform(
        [transaction_type]
    )[0]

    df = pd.DataFrame([{
        "step": step,
        "type": type_encoded,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud
    }])

    df = prepare_features(df)

    features = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balanceDiffOrig",
        "balanceDiffDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "isFlaggedFraud"
    ]

    X = df[features]

    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        X_scaled.shape[0],
        X_scaled.shape[1],
        1
    )

    prob = float(
        model.predict(
            X_scaled,
            verbose=0
        )[0][0]
    )

    prediction = (
        "🚨 FRAUD"
        if prob > 0.5
        else "✅ LEGITIMATE"
    )

    return (
        prediction,
        f"{prob * 100:.2f}%"
    )
    # =====================================
# BATCH CSV ANALYSIS
# =====================================

def analyze_csv(file):

    df = pd.read_csv(file.name)

    df["type"] = encoder.transform(df["type"])

    df = prepare_features(df)

    features = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balanceDiffOrig",
        "balanceDiffDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "isFlaggedFraud"
    ]

    X = df[features]

    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        X_scaled.shape[0],
        X_scaled.shape[1],
        1
    )

    probs = model.predict(
        X_scaled,
        verbose=0
    )

    probs = probs.flatten()

    df["Fraud_Probability"] = probs

    df["Prediction"] = np.where(
        probs > 0.5,
        "Fraud",
        "Legitimate"
    )

    fraud_count = (df["Prediction"] == "Fraud").sum()
    legit_count = (df["Prediction"] == "Legitimate").sum()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        [fraud_count, legit_count],
        labels=["Fraud", "Legitimate"],
        autopct="%1.1f%%"
    )

    ax.set_title("Fraud Detection Dashboard")

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    df.to_csv(
        output_file.name,
        index=False
    )

    summary = f"""
Total Transactions : {len(df)}

Fraud Transactions : {fraud_count}

Legitimate Transactions : {legit_count}
"""

    return (
        output_file.name,
        fig,
        summary
    )

# =====================================
# USER INTERFACE
# =====================================

with gr.Blocks() as demo:

    gr.Markdown(
        "# Smart Fraud Detection System"
    )

    with gr.Tab("Single Transaction Detection"):

        step = gr.Number(
            label="Step",
            value=0
        )

        transaction_type = gr.Dropdown(
            choices=[
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ],
            value="CASH_IN",
            label="Transaction Type"
        )

        amount = gr.Number(
            label="Amount",
            value=0
        )

        oldbalanceOrg = gr.Number(
            label="Old Balance Origin",
            value=0
        )

        newbalanceOrig = gr.Number(
            label="New Balance Origin",
            value=0
        )

        oldbalanceDest = gr.Number(
            label="Old Balance Destination",
            value=0
        )

        newbalanceDest = gr.Number(
            label="New Balance Destination",
            value=0
        )

        isFlaggedFraud = gr.Dropdown(
            choices=[0, 1],
            value=0,
            label="Flagged Fraud (0 or 1)"
        )

        predict_btn = gr.Button(
            "Predict Fraud"
        )

        prediction_output = gr.Textbox(
            label="Prediction"
        )

        confidence_output = gr.Textbox(
            label="Fraud Probability"
        )

        predict_btn.click(
            fn=predict_fraud,
            inputs=[
                step,
                transaction_type,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest,
                isFlaggedFraud
            ],
            outputs=[
                prediction_output,
                confidence_output
            ]
        )

    with gr.Tab("Batch CSV Detection"):

        csv_file = gr.File(
            label="Upload CSV File"
        )

        analyze_btn = gr.Button(
            "Analyze Dataset"
        )

        result_file = gr.File(
            label="Download Result CSV"
        )

        dashboard_plot = gr.Plot(
            label="Pie Chart"
        )

        summary_text = gr.Textbox(
            label="Dashboard Summary"
        )

        analyze_btn.click(
            fn=analyze_csv,
            inputs=csv_file,
            outputs=[
                result_file,
                dashboard_plot,
                summary_text
            ]
        )

# =====================================
# LAUNCH
# =====================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
