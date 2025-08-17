import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import img_to_array
from streamlit_lottie import st_lottie
import json

# Paths to the models
# Paths to the models
disease_model_paths = [
    "models/disease_models/cnn_model.h5",
    "models/disease_models/vgg16_model.h5",
    "models/disease_models/resnet50_model.h5",
    "models/disease_models/inceptionv3_model.h5",
    "models/disease_models/mobilenetv2_model.h5"
]
stage_model_path = "models/stage_model.h5"
treatment_data_path = "Treatment.xlsx"  # Ensure this file exists

img_height, img_width = 128, 128

disease_labels = ["Bacterial blight", "Blast", "Brown Spot", "Healthy Rice Leaf", "Leaf scald", "Tungro"]
stage_labels = ["2", "3", "4", "5"]

@st.cache_resource
def load_models():
    disease_models = [tf.keras.models.load_model(path, compile=False) for path in disease_model_paths]
    for model in disease_models:
        model.compile()
    stage_model = tf.keras.models.load_model(stage_model_path, compile=False)
    stage_model.compile()
    return disease_models, stage_model

@st.cache_resource
def load_treatment_data():
    return pd.read_excel(treatment_data_path)

disease_models, stage_model = load_models()
treatment_df = load_treatment_data()

def predict_disease_and_stage(image):
    image = image.resize((img_width, img_height))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    disease_predictions = [model.predict(img_array) for model in disease_models]
    avg_disease_prediction = np.mean(disease_predictions, axis=0)
    disease_class = disease_labels[np.argmax(avg_disease_prediction)]
    
    if disease_class.lower() == "healthy rice leaf":
        disease_class = "No Disease; it's a Healthy Leaf"
    
    stage_prediction = stage_model.predict(img_array)
    stage_class = stage_labels[np.argmax(stage_prediction)]
    
    return disease_class, stage_class

def recommend_treatment(disease, stage):
    treatment_row = treatment_df[(treatment_df["Disease"] == disease) & (treatment_df["Leaf Stage"] == int(stage))]
    if not treatment_row.empty:
        return treatment_row["Treatment"].values[0]
    else:
        return "No specific treatment found for this combination."

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.title("🌾 Paddy Leaf Disease Prediction and Treatment Recommendations")
lottie_animation = load_lottie_file("Animation - 1737090948465.json")
st_lottie(lottie_animation, height=300, key="paddy_leaf_animation")

uploaded_file = st.file_uploader("📷 Upload a Paddy Leaf Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("🔍 Predict"):
        disease, stage = predict_disease_and_stage(image)
        st.session_state.disease = disease
        st.session_state.stage = stage
        st.write(f"**🌱 Leaf Stage:** {stage}")
        st.write(f"**🦠 Disease:** {disease}")

# Show treatment recommendation only if disease prediction exists
if "disease" in st.session_state and st.session_state.disease != "No Disease; it's a Healthy Leaf":
    if st.button("💊 Show Treatment Recommendation"):
        treatment = recommend_treatment(st.session_state.disease, st.session_state.stage)

        # Display disease, stage, and treatment
        st.write(f"**🌱 Leaf Stage:** {st.session_state.stage}")
        st.write(f"**🦠 Disease:** {st.session_state.disease}")
        st.write(f"**📝 Recommended Treatment:** {treatment}")