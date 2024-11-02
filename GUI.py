import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from datetime import datetime

model = load_model('Detection.keras')
st.title("Age, Gender & Birthyear Detector")
st.write("Upload an image to detect age, gender, and estimated date of birth.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def Detect(image):
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    image = np.array([image]) / 255
    pred = model.predict(image)
    age = int(np.round(pred[1][0]))
    sex_f = ["Male", "Female"]
    sex = int(np.round(pred[0][0]))
    return age, sex_f[sex]

def calculate_birth_year(age):
    current_year = datetime.now().year
    birth_year = current_year - age
    return birth_year

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Predict age, gender, and date of birth
    if st.button('Detect'):
        age, gender = Detect(image)
        birth_year = calculate_birth_year(age)
        st.write(f"Predicted Age: {age}")
        st.write(f"Predicted Gender: {gender}")
        st.write(f"Estimated Year of Birth: {birth_year}")
