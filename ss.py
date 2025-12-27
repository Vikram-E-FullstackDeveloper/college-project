# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostRegressor

# -------------------------
# Config
# -------------------------
MODEL_DIR = "models"
FERT_JSON_FILE = "fertilizer_recommendationnew.json"

# -------------------------
# Load model and scaler
# -------------------------
@st.cache_resource
def load_model_artifacts():
    reg_model = CatBoostRegressor()
    reg_model.load_model(f"{MODEL_DIR}/fertilizer_quantity_model.cbm")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    return reg_model, scaler, encoders

reg_model, scaler, encoders = load_model_artifacts()

# -------------------------
# Load fertilizer JSON
# -------------------------
with open(FERT_JSON_FILE, "r") as f:
    fert_json = json.load(f)

# Normalize keys
def normalize_crop_name(name):
    return name.strip().replace(" ", "_").replace("-", "_").lower()

fert_json = {normalize_crop_name(k): v for k, v in fert_json.items()}

# -------------------------
# Preprocess input
# -------------------------
def preprocess_input(df, encoders, scaler):
    df_copy = df.copy()
    for col, le in encoders.items():
        if col in df_copy.columns:
            try:
                df_copy[col] = le.transform(df_copy[col].astype(str))
            except:
                df_copy[col] = 0
    # Ensure same columns as scaler
    missing_cols = set(scaler.feature_names_in_) - set(df_copy.columns)
    for c in missing_cols:
        df_copy[c] = 0
    df_copy = df_copy[scaler.feature_names_in_]
    scaled = scaler.transform(df_copy)
    return pd.DataFrame(scaled, columns=df_copy.columns)

# -------------------------
# Function to calculate dynamic nutrient amounts
# -------------------------
def dynamic_fertilizer_amount(crop_name, area, production):
    # Base fertilizer per ha (can be customized per crop)
    base_nutrients = fert_json.get(normalize_crop_name(crop_name), {
        "Nutrients": {
            "Nitrogen": {"Fertilizer": "Urea", "Recommended_Amount_kg_per_hectare": 100, "Application_Timing": "Baseline"},
            "Phosphorus": {"Fertilizer": "DAP", "Recommended_Amount_kg_per_hectare": 50, "Application_Timing": "Baseline"},
            "Potassium": {"Fertilizer": "MOP", "Recommended_Amount_kg_per_hectare": 50, "Application_Timing": "Baseline"},
        }
    })["Nutrients"]

    # Scale amount by area and production
    factor = area / 100 * production / 1000  # simple scaling factor
    for nutrient, val in base_nutrients.items():
        val["Recommended_Amount_kg_per_hectare"] = round(val["Recommended_Amount_kg_per_hectare"] * factor, 2)
    return base_nutrients

# -------------------------
# Streamlit UI
# -------------------------
st.title("🌾 Crop Fertilizer Recommendation System (CSV Batch Prediction)")

uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])
if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    
    # Preprocess
    input_scaled_df = preprocess_input(test_df, encoders, scaler)
    
    # Predict quantity
    test_df["Predicted_Quantity"] = reg_model.predict(input_scaled_df)
    test_df["Predicted_Quantity"] = test_df["Predicted_Quantity"].apply(lambda x: max(0, x))  # avoid negatives
    
    # Assign fertilizer dynamically
    fert_columns = []
    nutrient_tables = []
    for idx, row in test_df.iterrows():
        crop = row["Crop"]
        area = row.get("Area", 100)
        production = row.get("Production", 1000)
        nutrients = dynamic_fertilizer_amount(crop, area, production)
        fert_columns.append(", ".join([val["Fertilizer"] for val in nutrients.values()]))
        # Create table string
        table_str = pd.DataFrame([
            {"Nutrient": k, 
             "Fertilizer": v["Fertilizer"], 
             "Amount (kg/ha)": v["Recommended_Amount_kg_per_hectare"], 
             "Timing": v.get("Application_Timing","")} 
            for k,v in nutrients.items()
        ])
        nutrient_tables.append(table_str)
    
    test_df["Fertilizer_Names"] = fert_columns
    
    # Show results
    st.subheader("✅ Predicted Fertilizer Quantities and Types")
    st.dataframe(test_df)

    st.subheader("📖 Detailed Nutrient Table per Crop")
    for crop, table in zip(test_df["Crop"], nutrient_tables):
        st.markdown(f"**{crop}**")
        st.table(table)
