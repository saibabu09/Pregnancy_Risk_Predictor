
import gradio as gr
import pandas as pd
import numpy as np
import joblib

## Load model nad preprocessor
model = joblib.load("model.pkl")
processsor = joblib.load("processor.pkl")
feature_names = processsor.get_feature_names_out()

## Features mapping for summary
feature_mapping = {
    'Age': 'Higher age',
    'BMI': 'Unhealthy BMI',
    'Blood_Pressure_High': 'High blood pressure',
    'Glucose_Level_High': 'High glucose level',
    'Smoker_Yes': 'Smoking habit',
    'Alcohol_Intake_Yes': 'Alcohol consumption',
    'Thyroid_Disorder_Yes': 'Thyroid disorder',
    'Hemoglobin_Level': 'Low hemoglobin',
    'Fetal_Heart_Rate_High': 'Abnormal fetal heart rate',
    'Previous_Miscarriages': 'History of miscarriage',
    'Physical_Activity_Level_Low': 'Low physical activity',
    'Family_History_Of_Miscarriage_Yes': 'Family history of miscarriage'
}

def predict_preganancy_risk(
    Age, BMI, Blood_Pressure, Glucose_Level, Previous_Miscarriages, First_Pregnancy,
    Gravidity, Parity, Thyroid_Disorder, Smoker, Alcohol_Intake, Hemoglobin_Level,
    Physical_Activity_Level, Pregnancy_Interval, Fetal_Heart_Rate, Family_History_Of_Miscarriage):

  ## Prepare inputs
  input_data = pd.DataFrame([{
       "Age": Age,
        "BMI": BMI,
        "Blood_Pressure": Blood_Pressure,
        "Glucose_Level": Glucose_Level,
        "Previous_Miscarriages": Previous_Miscarriages,
        "First_Pregnancy": First_Pregnancy,
        "Gravidity": Gravidity,
        "Parity": Parity,
        "Thyroid_Disorder": Thyroid_Disorder,
        "Smoker": Smoker,
        "Alcohol_Intake": Alcohol_Intake,
        "Hemoglobin_Level": Hemoglobin_Level,
        "Physical_Activity_Level": Physical_Activity_Level,
        "Pregnancy_Interval": Pregnancy_Interval,
        "Fetal_Heart_Rate": Fetal_Heart_Rate,
        "Family_History_Of_Miscarriage": Family_History_Of_Miscarriage
  }])
  try:

    ### Preprocessing
    transformed = processsor.transform(input_data)
    prediction = model.predict(transformed)[0]

    ## generate summary if high risk
    summary = ""
    if prediction == 1 and hasattr(model, "coef_"):
      contrib = transformed[0] * model.coef_[0]
      feature_contribs = list(zip(feature_names, contrib))
      top_features = sorted(feature_contribs, key=lambda x: abs(x[1]), reverse=True)[:3]
      summary_factors = [
          feature_mapping.get(feat.split("_")[0], feat) for feat, _ in top_features
      ]
      summary = "

 🔎 Rsik Indicators: " + ",".join(summary_factors)

    return "✅ Low Risk" if prediction == 0 else "❌ High Risk"+summary
  except Exception as e:
    return f"❌ predition error: {str(e)}"


## Gradio Interface
interface = gr.Interface(
    fn=predict_preganancy_risk,
    inputs=[
        gr.Slider(18,45, label="Age"),
        gr.Slider(18, 40, label="BMI"),
        gr.Dropdown(["Normal","Elevated","High"], label="Blood Pressure"),
        gr.Dropdown(["Low","Normal","High"], label="Glucose Level"),
        gr.Slider(0,5, step=1, label="Previous Miscarriages"),
        gr.Radio(["Yes","No"], label="First Pregnancy"),
        gr.Slider(0,5, step=1, label="Gravidity"),
        gr.Slider(0,4,step=1, label="Parity"),
        gr.Radio(["Yes","No"], label="Thyroid Disorder"),
        gr.Radio(["Yes","No"], label="Smoker"),
        gr.Radio(["Yes","No"], label="Alcohol Intaker"),
        gr.Slider(8,15, label="Hemoglobin Level"),
        gr.Dropdown(["Low","Moderate","High"],label="physical Activity Level"),
        gr.Slider(0,10, step=1, label="Pregnancy Interval"),
        gr.Dropdown(["Normal","Medium","High"], label="Fetal Heart Rate"),
        gr.Radio(["Yes","No"], label="Family History Of Miscarriage")
    ],

    outputs = "text",
    title = "Pregnancy Risk Prediction",
    description = "Enter patient details to check pregnancy risk and understand contributing factors"
)

interface.launch()
