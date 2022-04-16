import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
import io
import os
from PIL import Image
import torch
from torchvision import transforms
import wget


def load_image():
     uploaded_file = st.file_uploader(label='Pick a traffic sign to predict')
     if uploaded_file is not None:
          image_data = uploaded_file.getvalue()
          st.image(image_data)
          return Image.open(io.BytesIO(image_data))
     else:
        return None

def load_model(path):
     model = models.load_model(path)

def main():
    st.title('TRAFFIC SIGN RECOGNITION')
    st.info('Student project for Maths 4997 @LSU')
    load_image()


if __name__ == '__main__':
    main()