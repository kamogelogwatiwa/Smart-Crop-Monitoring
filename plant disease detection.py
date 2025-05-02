import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pre-trained models
disease_model = load_model("crop_disease_model.keras")  # Keras binary classifier

st.set_page_config(page_title="AgriConnectAI: Crop Disease Detection", layout="wide")
st.title("🌾 AgriConnectAI: Smart Crop Monitoring System")

# Crop Disease Detection
st.header("🩺 Crop Disease Detection (Image-Based)")

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

