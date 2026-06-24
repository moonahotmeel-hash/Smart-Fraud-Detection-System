import gradio as gr
import pandas as pd
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "models/fraud_detection_model.keras"
)

with open(
    "models/scaler.pkl",
    "rb"
) as f:
    scaler = pickle.load(f)

with open(
    "models/label_encoder.pkl",
    "rb"
) as f:
    encoder = pickle.load(f)

# ==========================================
# GLOBAL DASHBOARD
# ==========================================

total_transactions = 0
fraud_transactions = 0
legitimate_transactions = 0

# ==========================================
# FEATURE ENGINEERING
# ==========================================

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

# ==========================================
# CREATE DASHBOARD CHART
# ==========================================

def create_dashboard_chart():

    labels = [
        "Legitimate",
        "Fraud"
    ]

    values = [
        legitimate_transactions,
        fraud_transactions
    ]

    plt.figure(figsize=(6,6))

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title(
        "Fraud Detection Statistics"
    )

    chart_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    plt.savefig(
        chart_file.name,
        bbox_inches="tight"
    )

    plt.close()

    return chart_file.name

# ==========================================
# GENERATE PDF REPORT
# ==========================================

def generate_pdf_report(
    result_text,
    confidence_text
):

    pdf_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(
        pdf_file.name
    )

    styles = getSampleStyleSheet()

    content = []

    try:

        content.append(
            Image(
                "assets/logo.png",
                width=120,
                height=120
            )
        )

    except:
        pass

    content.append(
        Paragraph(
            "University of Gedaref",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            "Faculty of Computer Science and Information Technology",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            "A Hybrid System Based on Deep Learning to Detect Financial Fraud by CNN-RNN (LSTM)",
            styles["Heading3"]
        )
    )

    content.append(
        Spacer(1,12)
    )

    content.append(
        Paragraph(
            f"Result: {result_text}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Confidence: {confidence_text}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Generated: {datetime.now()}",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return pdf_file.name
    # ==========================================
# SINGLE PREDICTION
# ==========================================

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

    global total_transactions
    global fraud_transactions
    global legitimate_transactions

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

    X = data[features]

    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        X_scaled.shape[0],
        X_scaled.shape[1],
        1
    )

    probability = float(
        model.predict(
            X_scaled,
            verbose=0
        )[0][0]
    )

    total_transactions += 1

    if probability > 0.5:

        fraud_transactions += 1

        result = "🔴 FRAUD TRANSACTION"

        message = (
            f"{result}\n\n"
            f"Confidence: {probability*100:.2f}%\n\n"
            f"Suspicious transaction detected."
        )

    else:

        legitimate_transactions += 1

        result = "🟢 LEGITIMATE TRANSACTION"

        message = (
            f"{result}\n\n"
            f"Confidence: {(1-probability)*100:.2f}%\n\n"
            f"Transaction appears normal."
        )

    fraud_rate = (
        fraud_transactions /
        total_transactions
    ) * 100

    dashboard = f"""
📊 DASHBOARD

Total Transactions: {total_transactions}

✅ Legitimate Transactions: {legitimate_transactions}

🚨 Fraud Transactions: {fraud_transactions}

📈 Fraud Rate: {fraud_rate:.2f}%
"""

    chart = create_dashboard_chart()

    confidence_text = (
        f"{probability*100:.2f}%"
    )

    return (
        message,
        confidence_text,
        dashboard,
        chart
    )

# ==========================================
# CLEAR FORM
# ==========================================

def clear_form():

    return (
        0,
        "CASH_IN",
        0,
        0,
        0,
        0,
        0,
        0,
        "",
        "",
        "",
        None
    )

# ==========================================
# ABOUT SYSTEM
# ==========================================

def about_system():

    return """
University of Gedaref

Faculty of Computer Science and Information Technology

Project:

A Hybrid System Based on Deep Learning to Detect Financial Fraud by CNN-RNN (LSTM)

Supervisor:
Hind Ali

Prepared By:

Ahmed Salah Eldin Abd Alrazig

Abo Ubaida Hamid

Amjad Ahmed

Ehab Alhaj

Mathani Adel

Batch 12

Information Systems Department
"""

# ==========================================
# CSV ANALYSIS
# ==========================================

def analyze_csv(file):

    df = pd.read_csv(
        file.name
    )

    df["type"] = encoder.transform(
        df["type"]
    )

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
    # ==========================================
# USER INTERFACE
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:

    try:
        gr.Image(
            "assets/university_logo.png",
            show_label=False,
            height=180
        )
    except:
        pass

    gr.Markdown("""
# Smart Fraud Detection System

### A Hybrid System Based on Deep Learning to Detect Financial Fraud by CNN-RNN (LSTM)

**University of Gedaref**  

""")

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
            choices=[0,1],
            value=0,
            label="Flagged Fraud (0 or 1)"
        )

        with gr.Row():

            predict_btn = gr.Button(
                "🔍 Predict Fraud",
                variant="primary"
            )

            clear_btn = gr.Button(
                "🧹 Clear Form"
            )

            print_btn = gr.Button(
                "🖨 Print Report"
            )

            about_btn = gr.Button(
                "ℹ About System"
            )

        prediction_output = gr.Textbox(
            label="Prediction",
            lines=5
        )

        confidence_output = gr.Textbox(
            label="Fraud Probability"
        )

        dashboard_output = gr.Textbox(
            label="Dashboard Statistics",
            lines=10
        )

        chart_output = gr.Image(
            label="Fraud Analytics"
        )

        about_output = gr.Textbox(
            label="About",
            lines=15
        )

        pdf_output = gr.File(
            label="Download Report"
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
                confidence_output,
                dashboard_output,
                chart_output
            ]
        )

        clear_btn.click(
            fn=clear_form,
            outputs=[
                step,
                transaction_type,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                oldbalanceDest,
                newbalanceDest,
                isFlaggedFraud,
                prediction_output,
                confidence_output,
                dashboard_output,
                chart_output
            ]
        )

        about_btn.click(
            fn=about_system,
            outputs=about_output
        )

        print_btn.click(
            fn=generate_pdf_report,
            inputs=[
                prediction_output,
                confidence_output
            ],
            outputs=pdf_output
        )

    with gr.Tab("Batch CSV Detection"):

        csv_file = gr.File(
            label="Upload CSV Dataset"
        )

        analyze_btn = gr.Button(
            "Analyze Dataset",
            variant="primary"
        )

        result_file = gr.File(
            label="Download Result CSV"
        )

        analyze_btn.click(
            fn=analyze_csv,
            inputs=csv_file,
            outputs=result_file
        )

# ==========================================
# START APP
# ==========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
