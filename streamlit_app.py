import random
import streamlit as st
import pandas as pd

st.write("""
# Image Caption Generator
Hello, we are Jonas, Moritz and Ole and our goal is to build, explain and present an image caption generator using Deep 
Learning and Neural Networks. We use these two subtypes of machine learning, as the combination of the two is closest 
to the way humans analyze images.
""")


@st.cache
def load_data():
    df = pd.read_csv("https://media.githubusercontent.com/media/buscjona/StreamLit-Image-Captioning"
                     "/main/sample_data_oneMill.csv")
    return df


data_load_state = st.subheader("Loading data...")
df = load_data()

data_load_state.subheader("Comparison generated caption vs. real caption:")


def get_image():
    column = random.randint(0, 999999)
    url = df.iloc[column]["URL"]
    st.image(url)
    st.write("real caption:")
    st.write(df.iloc[column]["TEXT"])
    st.write("URL:")
    st.write(url)


if st.button("Get a random image from the dataset."):
    get_image()
