import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform
import os

st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')

st.title("Reconocimiento de Imágenes")

# Imagen principal (encabezado) con validación
if os.path.exists('inicio.jpg'):
    imagen_inicio = Image.open('inicio.jpg')
    st.image(imagen_inicio, width=350)
else:
    st.warning("No se encontró 'inicio.jpg'")

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    img = Image.open(img_file_buffer).convert('RGB')  # ← IMPORTANTE
    img = img.resize((224, 224))

    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    prob_majo = prediction[0][0]

    if prob_majo > 0.5:
        st.header("Hola Majo")

        if os.path.exists('hello.png'):
            st.image('hello.png', width=300)
        else:
            st.warning("No se encontró 'hello.png'")

    else:
        st.header("No te veo majo")

        if os.path.exists('bye.png'):
            st.image('bye.png', width=300)
        else:
            st.warning("No se encontró 'bye.png'")
