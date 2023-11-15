# Import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Load Model
model = load_model('/Users/gloriapintado/Documents/Cervical-Cancer-Prediction/vgg16_model.h5')

# Title
st.title("Cervical Cancer Cell Type Prediction")


