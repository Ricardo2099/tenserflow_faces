import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# MIS PERSONAS (mismo orden que en entrenamiento)
listaPersonas = ['Adrian', 'Ander', 'Felix', 'Giovanni', 'Rafael', 'Roberto']

MODEL_PATH = os.path.join('.', 'reconocimiento-rostro', '1')
IMG_SIZE = (150, 150)

# ---- Cargar el modelo solo una vez ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Crear API ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocesar_imagen_bytes(file_bytes: bytes):
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("No se pudo leer la imagen")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 150, 150, 1)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # leer archivo
    file_bytes = await file.read()
    input_img = preprocesar_imagen_bytes(file_bytes)

    # predicción
    pred = model.predict(input_img)
    pred_idx = int(np.argmax(pred))
    prob = float(np.max(pred))

    persona = listaPersonas[pred_idx]

    # dict con probabilidades por persona
    probs = {listaPersonas[i]: float(pred[0][i]) for i in range(len(listaPersonas))}

    return {
        "prediccion": persona,
        "confianza": prob,
        "probabilidades": probs
    }
