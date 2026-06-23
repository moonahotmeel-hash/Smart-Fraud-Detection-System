import gradio as gr
import pandas as pd
import numpy as np
import pickle
import tempfile

from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

model = load_model("models/fraud_detection_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# =========================
# FEATURE ENGINEERING
# =========================

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


# =========================
# SINGLE PREDICTION
# =========================

def predict_fraud(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    balanceDiffOrig,
    balanceDiffDest,
    errorBalanceOrig,
    errorBalanceDest,
    isFlaggedFraud
):

    type_map = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4
    }

    data = [[
        step,
        type_map[transaction_type],
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        balanceDiffOrig,
        balanceDiffDest,
        errorBalanceOrig,
        errorBalanceDest,
        isFlaggedFraud
    ]]

    data_scaled = scaler.transform(data)

    data_scaled = data_scaled.reshape(
        data_scaled.shape[0],
        data_scaled.shape[1],
        1
    )

    prob = float(model.predict(data_scaled, verbose=0)[0][0])

    prediction = "🚨 Fraud" if prob > 0.5 else "✅ Legitimate"

    return f"""
Prediction: {prediction}

Fraud Probability: {prob*100:.2f}%
"""

    type_encoded = encoder.transform(
        [transaction_type]
    )[0]

    data = pd.DataFrame([{
        "step": step,
        "type": type_encoded,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud
    }])

    data = prepare_features(data)

    features = [
        'step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'balanceDiffOrig',
        'balanceDiffDest',
        'errorBalanceOrig',
        'errorBalanceDest',
        'isFlaggedFraud'
    ]

    X = data[features]

    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        X_scaled.shape[0],
        X_scaled.shape[1],
        1
    )

    prob = float(model.predict(X_scaled, verbose=0)[0][0])

    prediction = "🚨 FRAUD" if prob > 0.5 else "✅ LEGITIMATE"

    return (
        prediction,
        f"{prob*100:.2f}%"
    )


# =========================
# BATCH CSV PREDICTION
# =========================

def analyze_csv(file):

    df = pd.read_csv(file.name)

    df["type"] = encoder.transform(df["type"])

    df = prepare_features(df)

    features = [
        'step',
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'balanceDiffOrig',
        'balanceDiffDest',
        'errorBalanceOrig',
        'errorBalanceDest',
        'isFlaggedFraud'
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

    df["Fraud_Probability"] = probs

    df["Prediction"] = np.where(
        probs > 0.5,
        "Fraud",
        "Legitimate"
    )

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    df.to_csv(
        output_file.name,
        index=False
    )

    return output_file.name


# =========================
# UI
# =========================

with gr.Blocks() as demo:

    gr.Markdown(
        "# Smart Fraud Detection System"
    )

    with gr.Tab("Single Transaction Detection"):

        step = gr.Number(label="Step")

        transaction_type = gr.Dropdown(
            choices=[
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ],
            label="Transaction Type"
        )

        amount = gr.Number(label="Amount")

        oldbalanceOrg = gr.Number(
            label="Old Balance Origin"
        )

        newbalanceOrig = gr.Number(
            label="New Balance Origin"
        )

        oldbalanceDest = gr.Number(
            label="Old Balance Destination"
        )

        newbalanceDest = gr.Number(
            label="New Balance Destination"
        )

        isFlaggedFraud = gr.Number(
            label="Flagged Fraud (0 or 1)",
            value=0
        )

        predict_btn = gr.Button(
            "Predict Fraud"
        )

        prediction_output = gr.Textbox(
            label="Prediction"
        )

        confidence_output = gr.Textbox(
            label="Confidence"
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
            label="Upload CSV"
        )

        analyze_btn = gr.Button(
            "Analyze Dataset"
        )

        result_file = gr.File(
            label="Download Results"
        )

        analyze_btn.click(
            fn=analyze_csv,
            inputs=csv_file,
            outputs=result_file
        )

demo.launch(
    server_name="0.0.0.0"
)
