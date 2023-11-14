# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Ko94x7PmeGGXpwXRrJqIK6dWW_Tq9Xh
"""

from google.colab import drive
drive.mount('/content/drive')

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

vgg16_model = load_model('drive/MyDrive/Cervical_Cancer_Models/models/vgg16_model.h5')

st.title('Cervical Cancer Classification App')

uploaded_file = st.file_uploader("Choose a cervical image.....", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    processed_img = preprocess_image(img_array)

    # Make predictions
    predictions = vgg16_model.predict(processed_img)
    predicted_class = np.argmax(predictions)

    # Display the result
    st.write(f'Prediction: {class_names[predicted_class]}')
