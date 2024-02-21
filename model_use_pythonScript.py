# -*- coding: utf-8 -*-


import cv2
import tensorflow as tf
import numpy as np

# path of the model
model = tf.keras.models.load_model("model path")

def check(path):
  img = cv2.imread(path)
  resize = tf.image.resize(img,(128,128))
  scaled = resize/255
  np.expand_dims(scaled,0).shape
  img_new = np.expand_dims(scaled,0)
  y_hat = model.predict(img_new)
  if y_hat>=0.5:
    print(f"{y_hat} : DOG")
  else:
    print(f"{y_hat} : CAT")

check("Image Path")

