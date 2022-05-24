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
    st.write("""Für unser Projekt benutzen wir den LAION Datensatz, welcher uns als CSV-Datei zur Verfügung steht. Es 
    handelt sich hierbei um den derzeit größten frei zugänglichen Bild-Text-Datensatz der Welt (240TB). Die CSV-Datei 
    beinhaltet verschiedene Attribute, beispielsweise URL, TEXT, NSFW oder similarity. Über die URL können die 
    zahlreichen Biler geladen und dargestellt werden.
    Der Datensatz bildet die Grundlage für unser Ziel des Generierens von Bildbeschreibungen. Natürlich ist er in 
    seiner rohen Form viel zu groß, weshalb wir uns dazu entschlossen haben ein kleineres Sample benutzen. Auf eine 
    bestimmte Größe des Samples haben wir uns allerdings noch nicht festgelegt, da wir gerade noch dabei sind den 
    Datensatz zu filtern, um uns mit dem Image Caption Generator auf ein spezifisches Thema zu fokussieren. An dieser 
    Stelle könnt ihr uns gerne einen Tipp geben, wie ihr den Datensatz am besten nach einem Bereich filtern würdet.
    Eine Problematik unseres Projekts könnte das Vergleichen der Bildbeschreibungen sein, da manche Bildbeschreibungen
    sehr speziell sind und nicht immer den tatsächlichen Inhalt der Bilder wiedergeben.""")

with st.expander("2. Source Selection"):
    st.write("""Wie bereits in dem vorherigen Schritt erläutert, greifen wir auf den LAION Datensatz zurück. Die 
    richtige Datenmenge zum Trainieren, Validieren und Testen werden wir in den nächsten Tagen festlegen, um mit dem 
    Projekt zu starten. Um fehlende Datenquellen machen wir uns bei diesem Datensatz keine Gedanken.""")

with st.expander("3. Data Cleaning"):
    st.write("""Beschädigte Daten sind in unserem Datensatz nicht zu finden. In seltenen Fällen kann ein Bild nicht 
    geladen werden. Gegen dieses Problem ist unserer Ansicht nach aber nichts zu machen. Wie ebenfalls in "Data
    Understaning" erläutert, sind wie grade noch in dem Prozess unseren Datensatz zu filtern und irrelevante Bilder zu 
    entfernen.""")

with st.expander("4. Feature Engineering"):
    st.write("""""")

with st.expander("5. Data Splitting"):
    st.write(""".""")
st.write("")


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
    st.write("Real caption: None")


if st.button("Get a random image from the dataset"):
    get_image()
