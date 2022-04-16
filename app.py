import streamlit as st

st.title("Traffic Sign Recognition")

st.write("MATHS 4997 PROJECT")
st.info("TRAFFIC SIGN RECOGNITION")

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)


picture = st.camera_input("Take a picture")

if picture:
     st.image(picture)