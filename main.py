import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import keras.utils as image
from PIL import Image
import streamlit as st
import io

def set_bg_hackdisease_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpaperbat.com/img/146005-ultra-hd-macro-wallpaper-top-free-ultra-hd-macro-background.jpg");
             background-size: cover;
         }}
         .main-container {{
             text-align: center;
             padding: 20px;
         }}
         .prediction-box {{
             background-color: #2a2a2a;
             color: white;
             padding: 15px;
             font-size: 18px;
             border-radius: 10px;
             text-align: center;
             font-weight: bold;
             margin-top: 20px;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache_resource(ttl=48*3600)
def load_model():
    return keras.models.load_model('model.h5', compile=False)

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

def predict_disease(model, img_array):
    labels = ['Pepper Bell - Bacterial Spot', 'Pepper Bell - Healthy', 'Potato - Early Blight',
              'Potato - Late Blight', 'Potato - Healthy', 'Tomato - Early Blight',
              'Tomato - Late Blight', 'Tomato - Healthy']
    probabilities = model.predict(img_array)
    return labels[np.argmax(probabilities)]

def main():
    set_bg_hackdisease_url()
    model = load_model()
    st.title('🌱 Plant Disease Detection')
    st.write("### Upload a leaf image to detect potential diseases.")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    image_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], label_visibility='collapsed')
    
    if image_file:
        img = Image.open(image_file)
        w, h = img.size
        w = 600 if w > h else int(w * (600.0 / h))
        st.image(img, width=w)
        
        if st.button("🔍 Analyze Image"):
            with st.spinner("Processing... Please wait."):
                img_array = preprocess_image(img)
                prediction = predict_disease(model, img_array)
            
            st.markdown(f'<div class="prediction-box">🩺 Prediction: {prediction}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
