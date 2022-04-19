import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from skimage import transform
from skimage import exposure
# from skimage import io
import os
import numpy as np
from PIL import Image
import io



def load_image():
     uploaded_file = st.file_uploader(label='Pick a traffic sign to predict')
     if uploaded_file is not None:
          image_data = uploaded_file.getvalue()
          image = Image.open(io.BytesIO(image_data))
          st.image(image_data)
          return np.asarray(image)
          #return Image.open(image_data)

     else:
        return None

def load_model(path):
     model = models.load_model(path)

def prepare_image(image):
     image = transform.resize(image, (32, 32))
     image = exposure.equalize_adapthist(image, clip_limit=0.1)
     image = image.astype("float32") / 255.0
     return image.reshape(1, 32,32,3)

def predict_image(image, model):
    return np.array(model(image)).round(3)

def main():
     st.title('TRAFFIC SIGN RECOGNITION')
     st.info('Student project for Maths 4997 @LSU')
     image  = load_image()
     model = load_model("trafficNet.h5")
     result = st.button('Predict Image')
     if result:
          st.write('Calculating results...')
          test_image = prepare_image(image)
          prediction = predict_image(test_image, model=model)
          print(prediction)
    
if __name__ == '__main__':
    main()