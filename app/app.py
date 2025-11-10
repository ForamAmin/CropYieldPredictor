import streamlit as st
import pandas as pd
import joblib
import os

# --- Page setup (MUST be the first st command) ---
st.set_page_config(page_title="Crop Yield Predictor 🌾", page_icon="🌿", layout="centered")

# --- Define Correct Model Paths ---
# Get the absolute path to the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to your models by going UP one level (..) 
# and then into the 'models' directory.
COARSE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'coarse_model.joblib')
RICE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'rice_model.joblib')
SOYBEAN_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'soybean_model.joblib')
WHEAT_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'wheat_model.joblib')

# Create a dictionary to map crop names to their correct file paths
MODEL_PATHS = {
    "wheat": WHEAT_MODEL_PATH,
    "rice": RICE_MODEL_PATH,
    "coarse": COARSE_MODEL_PATH,
    "soybean": SOYBEAN_MODEL_PATH
}

# --- App UI Starts Here ---
st.title("🌾 Smart Crop Yield Predictor")
st.markdown("### Predict how crop yields change under different climate conditions.")
st.write("Select crop type, climate model, adaptation level, and CO₂ level to estimate yield percentage change.")

# --- Crop Selection ---
crop = st.selectbox(
    "Select Crop Type",
    ["wheat", "rice", "coarse", "soybean"],
    index=0
)

# --- Other Inputs ---
col1, col2 = st.columns(2)

with col1:
    climate_model = st.selectbox("Climate Model", ["GISS", "GFDL", "UKMO", "HadCM3"])
    adaptation = st.radio("Adaptation Level", ["none", "level1", "level2"])
with col2:
    region = st.selectbox("Region", ["Asia", "Europe", "Africa"])
    co2 = st.slider("CO₂ ppm Level", 300, 600, 400)

# --- Predict Button ---
if st.button("🔮 Predict Yield Change"):
    
    # --- 1. Load correct model based on selected crop ---
    # Get the correct, safe path from the dictionary
    selected_model_path = MODEL_PATHS[crop] 

    try:
        # Load the model using the CORRECT path
        model = joblib.load(selected_model_path)
        
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Model file not found. Please ensure the model exists at: {selected_model_path}")
        st.stop() # Stop the script if the model isn't found
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        st.stop()

    # --- 2. Create DataFrame for prediction ---
    df = pd.DataFrame({
        "climate_model": [climate_model],
        "co2": [co2],
        "adaptation": [adaptation],
        "region": [region]
    })

    # --- 3. Encode categories (WARNING: See note below) ---
    # This method is fragile and assumes the order in the selectbox
    # perfectly matches the order the model was trained on.
    for col in ["climate_model", "adaptation", "region"]:
        df[col] = df[col].astype("category").cat.codes

    # --- 4. Predict ---
    try:
        prediction = model.predict(df)[0]

        # --- 5. Display result ---
        if prediction >= 0:
            st.success(f"🌱 Predicted Yield Increase: **+{prediction:.2f}%**")
        else:
            st.error(f"🍂 Predicted Yield Decrease: **{prediction:.2f}%**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("This often happens if the input data (like 'GISS') doesn't match what the model was trained on.")


# --- Footer ---
st.markdown("---")
st.caption("Developed by Foram 🌿 | Machine Learning Project | 2025")