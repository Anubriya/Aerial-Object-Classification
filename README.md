# Aerial-Object-Classification
This project focuses on classifying Birds and Drones from aerial images using Deep Learning and Transfer Learning techniques.   The best performing model is deployed using a Streamlit web application, allowing users to upload images and instantly receive predictions along with a confidence score.

## 🚀 Project Overview

With increasing drone usage, distinguishing between natural aerial objects (birds) and artificial flying machines (drones) has become important.  
This project builds multiple deep learning models and compares their performance:

- 🧱 **Custom CNN**
- 🟦 **MobileNetV2 (Best Performing)**
- 🟥 **ResNet50**
- 🟩 **EfficientNetB0**

After evaluation, **MobileNetV2** achieved the highest accuracy and became the final deployed model.

## 🧪 Model Training

### 🔹 Preprocessing
- Resized all images to **224 × 224**
- Normalized pixel values (0–1)
- Applied data augmentation on training set

### 🔹 Models Used
1. **Custom CNN**
2. **EfficientNetB0**
3. **MobileNetV2**
4. **ResNet50**

### 🔹 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Visualization of Predictions  

---

## 🌐 Streamlit Web App

The Streamlit app allows users to:

✔ Upload an image (JPG/PNG)  
✔ Preview the uploaded image  
✔ Get prediction: **Bird** or **Drone**  
✔ View the **confidence score** (model certainty)  

### Run the app locally:

py -3.10 -m streamlit run aerial.py


Ensure that:

- Python 3.10 is installed  
- TensorFlow, Streamlit, Pillow, NumPy are installed  

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV / Pillow
- Streamlit
- NumPy / Pandas
- Matplotlib / Seaborn

## 👩‍💻 Author

**Anubriya Baskaran**  



