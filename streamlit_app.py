import streamlit as st
import pandas as pd

st.title("Image Caption Generator")
st.text("Hello, we are Jonas, Moritz and Ole and our goal is to build, explain and present")
st.text("an image caption generator using Deep Learning and Neural Networks. We use these")
st.text("two subtypes of machine learning, as the combination of the two is closest to ")
st.text("the way humans analyze images.")
st.text("")

data_load_state = st.subheader("Loading data...")
data = pd.read_csv("https://github.com/buscjona/StreamLit-Image-Captioning/blob/main/sample_data_oneMill.csv")
data_load_state.subheader("Loading data...done!")
