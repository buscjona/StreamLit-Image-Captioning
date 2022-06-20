import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import random
import streamlit as st


#load data
features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_14.h5')
images = "Bilder/"
max_length = 33
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))


#generate captions
def image_captioning(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

#show picture an caption
def show():
    z = random.randint(0,4)
    pic = list(features.keys())[z]
    image = features[pic].reshape((1,2048))
    x = plt.imread(images + pic)
    st.image(x)
    st.write("Caption:", image_captioning(image))
    
if st.button("Generate caption"):
    show()
    
