import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("best_model_mobilenet.keras")

st.set_page_config(page_title="Bird vs Drone Classifier", page_icon="🛩️")

# ---------------------- UI Title ----------------------
st.title("🛩️ Bird vs Drone Classification")
st.write("""
Upload an image and the model will classify whether it is a **Bird** or a **Drone**.  
The confidence score shows how sure the model is about its prediction.
""")

# ---------------------- File Upload ----------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]

    # Calculate confidence score
    confidence = float(pred) if pred > 0.5 else float(1 - pred)

    # Final label
    label = "Drone" if pred > 0.5 else "Bird"

    # ---------------------- Result Box ----------------------
    st.subheader("🔍 Prediction Result")
    st.write(f"### **Prediction: {label}**")
    st.write(f"### **Confidence Score: {confidence * 100:.2f}%**")

    
