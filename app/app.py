import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------------------------------------------------
# Model & Preprocessor Loading
# ---------------------------------------------------------------------
# Get the absolute path of the current script
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(APP_DIR)
# Define the models directory path
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# These names now match your .pkl files
MODEL_FILES = {
    'Wheat': 'wheat.pkl',
    'Rice': 'rice.pkl',
    'Coarse Grains': 'coarse grains.pkl',
    'Grains': 'grains.pkl',
    'Protein Feed': 'protein_feed.pkl',
    'Four Commodities': 'four commo-dities.pkl'
}

@st.cache_resource
def load_assets():
    """
    Loads the preprocessor and all models into memory.
    """
    try:
        # Load the single preprocessing pipeline
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        st.error(f"Error: 'preprocessor.pkl' not found at {preprocessor_path}")
        st.error("Honest check: Have you moved 'preprocessor.pkl' from 'notebooks/' to 'models/'?")
        return None, None
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None, None

    # Load all models and map them to the dictionary
    models = {}
    for crop_name, file_name in MODEL_FILES.items():
        try:
            model_path = os.path.join(MODELS_DIR, file_name)
            models[crop_name] = joblib.load(model_path)
        except FileNotFoundError:
            st.warning(f"Warning: Model file '{file_name}' not found. Skipping '{crop_name}'.")
        except Exception as e:
            st.error(f"Error loading model {file_name}: {e}")
            
    return preprocessor, models

# Load assets on script run
PREPROCESSOR, MODELS = load_assets()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🌾 Crop Yield Prediction")
st.markdown("Enter the required parameters to get a yield prediction.")
st.markdown("---")

# ---------------------------------------------------------------------
# Sidebar - User Inputs
# ---------------------------------------------------------------------
st.sidebar.header("Prediction Inputs")
st.sidebar.caption("Provide the values for the 6 features the model was trained on.")

# --- 1. Target Crop ---
selected_crop = st.sidebar.selectbox(
    "1. Select Crop to Predict",
    options=list(MODELS.keys()),
    help="Choose the target crop you want to predict the yield for."
)

# --- 2. BLS Code ---
bls_code = st.sidebar.number_input(
    "2. BLS Code",
    min_value=0,
    value=913,
    step=1,
    help="Enter the single BLS code (e.g., 913, 101, 777). This will be converted to a string for the preprocessor."
)

# --- 3. Scenario ---
scenario = st.sidebar.selectbox(
    "3. Scenario",
    options=['CM2-S550', 'CM2-S750', 'CM3-A', 'GFDL', 'GISS', 'UKMO'],
    help="Select the climate scenario."
)

# --- 4. Time_Slice ---
time_slice = st.sidebar.number_input(
    "4. Time Slice (Year)",
    min_value=1980,
    max_value=2150, # Set max to 2150 to allow for 'Equilibrium'
    value=2025,
    step=1,
    help="Enter the specific year (e.g., 2025). This will be converted to a float."
)

# --- 5. CO2 effects (FIXED: Changed from checkbox to selectbox) ---
co2_effects = st.sidebar.selectbox(
    "5. CO2 Effects",
    options=['EquilibriuYes', 'EquilibriuNo'], # From your final.xlsx
    help="Select the CO2 effect status."
)

# --- 6. CO2 ppm ---
co2_ppm = st.sidebar.number_input(
    "6. CO2 ppm",
    min_value=0.0,
    max_value=2000.0,
    value=450.0,
    step=1.0,
    help="Enter the CO2 concentration in parts-per-million."
)

# --- 7. Adaptation (FIXED: Added 'Yes' to options) ---
adaptation = st.sidebar.selectbox(
    "7. Adaptation",
    options=['No', 'Yes', 'Level 1', 'Level 2'], # From your final.xlsx
    help="Select the adaptation measure."
)

# --- Prediction Button ---
predict_button = st.sidebar.button("Predict Yield")

# ---------------------------------------------------------------------
# Main Page - Prediction Logic & Display
# ---------------------------------------------------------------------
if predict_button:
    if PREPROCESSOR is None or not MODELS:
        st.error("Models or preprocessor could not be loaded. Please check file paths and errors.")
    else:
        try:
            # 1. Create a 1-row DataFrame from the inputs
            # The keys MUST match the names from INPUT_FEATURES
            input_data = {
                'BLS Code': [bls_code],
                'Scenario': [scenario],
                'Time_Slice': [time_slice],
                'CO2 effects': [co2_effects], # Use the direct string value
                'CO2 ppm': [co2_ppm],
                'Adaptation': [adaptation]
            }
            X_new_raw = pd.DataFrame(input_data)
            
            # --- FIX: Force correct data types *BEFORE* transforming ---
            # This is what solves the "ufunc 'isnan'" TypeError
            X_new_raw['BLS Code'] = X_new_raw['BLS Code'].astype(object) # Was categorical
            X_new_raw['Scenario'] = X_new_raw['Scenario'].astype(object) # Was categorical
            X_new_raw['Time_Slice'] = X_new_raw['Time_Slice'].astype(float) # Was numeric (float)
            X_new_raw['CO2 effects'] = X_new_raw['CO2 effects'].astype(object) # Was categorical
            X_new_raw['CO2 ppm'] = X_new_raw['CO2 ppm'].astype(float) # Was numeric
            X_new_raw['Adaptation'] = X_new_raw['Adaptation'].astype(object) # Was categorical
            
            st.subheader("1. Raw Input Data (Types Corrected)")
            st.dataframe(X_new_raw)
            # st.write(X_new_raw.dtypes) # Optional: uncomment to debug

            # 3. Process the raw data using the loaded preprocessor
            
            
            X_new_processed = PREPROCESSOR.transform(X_new_raw)
            
            
            # 4. Select the correct model based on user's dropdown
            model_to_use = MODELS[selected_crop]
            
            # 5. Make the prediction
            prediction_result = model_to_use.predict(X_new_processed)
            
            # 6. Display the result
            st.markdown("---")
            st.subheader(f"2. 📈 Prediction Result for: {selected_crop}")
            
            st.metric(
                label=f"Predicted {selected_crop} Yield",
                value=f"{prediction_result[0]:.2f}",
                help="Predicted yield (units are % change)."
            )

        except Exception as e:
            st.error(f"An error occurred during prediction:")
            st.exception(e)
            st.warning("Please ensure your input column names (e.g., 'BLS Code', 'Scenario') exactly match the names used during model training.")

else:
    st.info("Fill out the inputs in the sidebar and click 'Predict Yield'.")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("*A Streamlit GUI for the Crop Yield Prediction model.*")