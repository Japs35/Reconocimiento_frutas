# -*- coding: utf-8 -*-
"""Proyecto2_frutas_ejecucion_camara.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LzSUU_2uhNorhaW5PrDWtrlu3a2kYIxJ
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

from google.colab import drive
drive.mount('/content/drive',force_remount=True)

path_json="/content/drive/MyDrive/Redes Neuronales/Modelos entrenados/Frutas/frutas.json"
path_h5="/content/drive/MyDrive/Redes Neuronales/Modelos entrenados/Frutas/frutas.weights.h5"

json_file=open(path_json,'r')
modelo_json=json_file.read()
json_file.close()

modelo=tf.keras.models.model_from_json(modelo_json)
modelo.load_weights(path_h5)

print("Se cargó el modelo!!!")

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

#test

img_height=200
img_width=200

from tensorflow.keras.preprocessing import image

path_img_test=filename
img=  image.load_img(path_img_test,target_size=(img_height,img_width))

#Convierto la imagen en matriz
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
imagen=np.vstack([x])

#Abrimos la imagen
from google.colab.patches import cv2_imshow
img=cv2.imread(path_img_test)
res=cv2.resize(img,(100,100))
cv2_imshow(res)


y_pred=modelo.predict(imagen)
print(y_pred)

#####################################

print("{}% de probabilidad que sea manzana".format(y_pred[0][0]*100))
print("{}% de probabilidad que sea naranja".format(y_pred[0][1]*100))
print("{}% de probabilidad que sea banana".format(y_pred[0][2]*100))



