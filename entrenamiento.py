import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import base64  # ahora mismo no se usa, pero lo dejo si después lo necesita

# ---- CONFIGURACIÓN ----
images = []
labels = []
listaPersonas = ['Adrian', 'Ander', 'Felix', 'Giovanni', 'Rafael', 'Roberto']

dataPath = f'{os.getcwd()}/data'
size = (150, 150)

# Extensiones válidas de imagen
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

print(f"Buscando imágenes en: {dataPath}")
print(f"Personas: {listaPersonas}")

for nombrePersona in listaPersonas:
    rostrosPath = os.path.join(dataPath, nombrePersona)

    if not os.path.isdir(rostrosPath):
        print(f"[AVISO] No existe la carpeta: {rostrosPath}. Se omite esta persona.")
        continue

    count_persona = 0

    for filename in os.listdir(rostrosPath):
        # Solo tomar archivos de imagen
        if not filename.lower().endswith(valid_exts):
            print(f"[AVISO] Se omite (no es imagen): {os.path.join(rostrosPath, filename)}")
            continue

        img_path = os.path.join(rostrosPath, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[AVISO] No se pudo leer la imagen, se omite: {img_path}")
            continue

        img = cv2.resize(img, size)
        images.append(img)
        labels.append(nombrePersona)
        count_persona += 1

    print(f"{nombrePersona}: {count_persona} imágenes cargadas.")

if len(images) == 0:
    raise RuntimeError("No se cargaron imágenes. Revise las carpetas en la carpeta 'data'.")

# ---- PREPARAR DATOS ----
images = np.array(images, dtype=np.float32)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels_encoded,
    test_size=0.2,
    random_state=42
)

X_train = X_train / 255.0
X_test = X_test / 255.0

# ---- MODELO ----
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(listaPersonas), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Comenzando entrenamiento...")
model.fit(
    X_train.reshape(-1, 150, 150, 1),
    y_train,
    epochs=30,
    validation_data=(X_test.reshape(-1, 150, 150, 1), y_test)
)

# ---- GUARDAR MODELO ----
export_path = 'reconocimiento-rostro/1/'
os.makedirs(export_path, exist_ok=True)

tf.keras.models.save_model(model, os.path.join('./', export_path))
print(f"Modelo guardado en: {export_path}")
