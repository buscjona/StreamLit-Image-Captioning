import random
import streamlit as st
import pandas as pd

# Headline
st.write("# ML4B - Image Caption Generator")

# Project explanation
st.header("Short project explanation")
st.write(""" Hello, we are Jonas, Moritz and Ole. Together we have set ourselves the goal of building, explaining and 
presenting an image caption generator using Deep Learning and its Neural Networks. We use these subtype of machine 
learning, as it is the closest to the way humans analyze images. Based on this, we want to compare our own generated 
caption with the actual caption of the image. The tip of the iceberg would be if we manage to algorithmically 
evaluate the appropriateness of our caption.
""")


# Data understanding


# Load data
@st.cache
def load_data():
    dataframe = pd.read_csv("https://media.githubusercontent.com/media/buscjona/StreamLit-Image-Captioning"
                            "/main/sample_data_oneMill.csv")
    return dataframe


data_load_state = st.header("Loading data...")
df = load_data()

data_load_state.header("Generated caption vs. real caption:")


# Get image
def get_image():
    column = random.randint(0, 999999)
    url = df.iloc[column]["URL"]
    st.image(url)
    st.write("Real caption: " + df.iloc[column]["TEXT"])
    st.write("URL: " + url)


if st.button("Get a random image from the dataset."):
    get_image()
