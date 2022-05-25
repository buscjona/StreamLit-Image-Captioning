import random
import streamlit as st
import pandas as pd

# Headline
st.write("# ML4B - Image Caption Generator")

# Project explanation
st.header("Short project explanation")
st.write("""Hello, we are Jonas, Moritz and Ole. Together we have set ourselves the goal of building, explaining and 
presenting an image caption generator using Deep Learning and its Neural Networks. We use these subtype of machine 
learning, as it is the closest to the way humans analyze images. Based on this, we want to compare our own generated 
caption with the actual caption of the image. The tip of the iceberg would be if we manage to algorithmically 
evaluate the appropriateness of our caption.""")
st.write("")

# Data preparation
st.header("Our current data preparation process")

with st.expander("1. Data Understanding"):
    st.write("""For our project we are using the LAION data set, which is available to us as a CSV file. It is currently 
    the largest freely accessible image-text dataset in the world (240TB). The CSV file contains various attributes, 
    for example URL, TEXT, NSFW or similarity. The URL can be used to load numerous images and display them. 
    The data set sets the ground for our goal of generating image captions. Of course it is  
    too large in its actual form, which is why we decided to use a smaller sample. We agreed on a sample size of about 
    1,000 - 5,000 examples. However, we are still in the process of filtering the data set in order to focus on a specific 
    theme. One of the problems of our project could be the comparison of the image captions, because some of the captions 
    are very specific and do not always reflect the actual content of the images.""")

with st.expander("2. Source Selection"):
    st.write("""As already explained in the previous step, we make use of the LAION data set. The total data for 
    training, validating and testing will initially be 1,000 - 5,000 images to not make the process too time-consuming.
    We are not concerned about missing data sources with this data set.""")

with st.expander("3. Data Cleaning"):
    st.write("""Corrupted data cannot be found in our data set. In rare cases, an image cannot be be loaded. However, 
    in our opinion there is nothing to be done about this problem. As also explained in "Data Understaning", we are 
    still in the process of filtering and expanding our dataset.""")

with st.expander("4. Feature Engineering"):
    st.write("""In terms of feature engineering, we decided to use the typical techniques for images:
     Resizing, cropping, clipping, blur, etc. We are keeping other possibilities open in this area.""")

with st.expander("5. Data Splitting"):
    st.write("""For the time being, we would like to start with a total of about 5,000 images. These will be divided 
    into a Train Set (70-80%), Validation Set (10-15%) and Test Set (10-15%). The process of dividing will be random.""")
st.write("")


# Load data
@st.cache
def load_data():
    dataframe = pd.read_csv("https://raw.githubusercontent.com/buscjona/StreamLit-Image-Captioning/main/data%20subsets/subsets.csv")
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
    st.write("Real caption: None")


if st.button("Get a random image from the dataset"):
    get_image()
