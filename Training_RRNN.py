import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

path_json = "Frutas/frutas.json"
path_h5 = "Frutas/frutas.weights.h5"

json_file=open(path_json,'r')
modelo_json=json_file.read()
json_file.close()

modelo=tf.keras.models.model_from_json(modelo_json)
modelo.load_weights(path_h5)

print("Se cargó el modelo!!!")

# Dimensiones de la imagen que espera el modelo
img_height = 200
img_width = 200

# Etiquetas de las clases
etiquetas = ["manzana", "naranja", "banana"]

# Iniciar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara")
        break

    # Redimensionar el frame a las dimensiones esperadas por el modelo
    img_resized = cv2.resize(frame, (img_height, img_width))
    print("Frame redimensionado:", img_resized.shape)

    # Convertir la imagen en matriz y normalizarla
    img_array = np.expand_dims(img_resized / 255.0, axis=0)  # Normalizar a rango [0, 1]
    print("Array de la imagen:", img_array.shape)

    # Hacer la predicción
    y_pred = modelo.predict(img_array)
    print("Predicción:", y_pred)

    # Obtener la clase con mayor probabilidad
    clase_predicha = np.argmax(y_pred)

    confidence_threshold = 0.60

    # Verificar si la predicción supera el umbral de confianza
    if y_pred[0][clase_predicha] > confidence_threshold:
        # Crear la etiqueta si la confianza es suficiente
        label = "{}: {:.2f}%".format(etiquetas[clase_predicha], y_pred[0][clase_predicha] * 100)
    else:
        # Mostrar mensaje de baja confianza si no supera el umbral
        label = "Confianza insuficiente"

    # Mostrar la etiqueta en el frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar el frame en una ventana
    cv2.imshow('Detección de Frutas', frame)

    # Si presionas 'q', termina el ciclo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()