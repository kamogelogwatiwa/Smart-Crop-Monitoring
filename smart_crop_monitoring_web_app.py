import streamlit as st
import numpy as np
import requests
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pre-trained models
health_model = joblib.load("crop_health_model_with_selected_features.pkl")  # Scikit-learn model
disease_model = load_model("crop_disease_model.keras")  # Keras binary classifier

st.set_page_config(page_title="AgriConnectAI", layout="wide")
st.title("🌾 AgriConnectAI: Smart Crop Monitoring System")

# Tabs for prediction tasks
tab1, tab2 = st.tabs(["📊 Crop Health Prediction", "🩺 Crop Disease Detection"])

# -------------------- Crop Health Prediction Tab --------------------

with tab1:
    st.header("📊 Plant Health Prediction")

    input_mode = st.radio("Choose Input Mode:", ["Manual Entry", "IoT Sensor Feed"])

    if input_mode == "Manual Entry":
        st.subheader("Enter Sensor Data Manually")
        soil_moisture = st.slider("Soil Moisture (%)", 10.0, 40.0, 25.0)
        nitrogen_level = st.slider("Nitrogen Level (mg/kg)", 10.0, 50.0, 30.0)
        potassium_level = st.slider("Potassium Level (mg/kg)", 10.0, 50.0, 30.0)

    else:
        st.subheader("Fetching Real-Time Sensor Data from IoT Setup")

        # Replace this with your actual IoT endpoint
        IOT_DATA_URL = "https://example.com/api/iot-data"  # <-- Update this to your actual API

        try:
            response = requests.get(IOT_DATA_URL)
            response.raise_for_status()
            data = response.json()

            soil_moisture = float(data["Soil_Moisture"])
            nitrogen_level = float(data["Nitrogen_Level"])
            potassium_level = float(data["Potassium_Level"])

            st.success("✅ Real-time sensor data fetched successfully.")
            st.metric("Soil Moisture (%)", soil_moisture)
            st.metric("Nitrogen Level (mg/kg)", nitrogen_level)
            st.metric("Potassium Level (mg/kg)", potassium_level)

        except Exception as e:
            st.error(f"Failed to fetch sensor data: {e}")
            st.stop()

    # Model Prediction
    if st.button("🔍 Predict Crop Health"):
        label_map = {0: "Healthy", 1: "Moderate", 2: "Stressed"}
        input_array = np.array([[soil_moisture, nitrogen_level, potassium_level]])
        predicted_class = health_model.predict(input_array)[0]
        prediction = label_map[predicted_class]

    
        st.markdown(f"### 🧠 Predicted Crop Health Status: **{prediction}**")
    
        # --- RULE-BASED RECOMMENDATIONS ---
        st.markdown("### 🔎 Recommendation Based on Sensor Conditions")
    
        if prediction == "Healthy":
            if soil_moisture > 35:
                st.info("Soil moisture is high, monitor for waterlogging signs.")
            elif nitrogen_level > 45:
                st.info("Nitrogen is high, reduce fertilizer to prevent toxicity.")
            else:
                st.success("Plant is healthy. Maintain current irrigation and fertilization schedules.")
    
        elif prediction == "Moderate":
            if soil_moisture < 20 and nitrogen_level < 20:
                st.warning("Soil moisture and nitrogen are both low. Recommend watering and applying nitrogen fertilizer.")
            elif nitrogen_level < 20:
                st.warning("Low nitrogen. Apply nitrogen-based fertilizer.")
            elif potassium_level < 20:
                st.warning("Low potassium. Supplement potassium to improve plant resilience.")
            else:
                st.info("Moderate condition. Monitor and adjust irrigation or nutrients accordingly.")
    
        elif prediction == "Stressed":
            if soil_moisture < 15 and nitrogen_level < 15 and potassium_level < 15:
                st.error("Critical deficiency detected. Immediate irrigation and NPK fertilization required.")
            elif soil_moisture < 15:
                st.error("Severe drought stress. Irrigate immediately.")
            elif nitrogen_level < 15:
                st.error("Severe nitrogen deficiency. Apply urea or ammonium nitrate.")
            elif potassium_level < 15:
                st.error("Low potassium. Apply potash fertilizer.")
            else:
                st.warning("Plant under stress. Review environmental and soil conditions holistically.")

 

"""
# -------------------- Crop Disease Detection Tab --------------------
with tab2:
    st.header("🩺 Leaf Disease Detection (Image-Based)")

    uploaded_image = st.file_uploader("Upload a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Model Prediction
        prediction = disease_model.predict(img_array)
        disease_class = "Diseased" if prediction[0][0] > 0.5 else "Healthy"

        st.markdown(f"### 🔍 Prediction: **{disease_class}**")

        if disease_class == "Diseased":
            st.warning("⚠️ Disease detected. Isolate affected plant and consider applying treatment.")
        else:
            st.success("✅ Leaf appears healthy. Continue routine monitoring.")

"""
            

