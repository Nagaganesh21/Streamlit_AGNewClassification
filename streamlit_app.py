import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import mngdataclean as mng
import re
import numpy as np

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Load model
model = load_model('news_classification.h5')

# Load max_len
max_len = pickle.load(open("max_len.pkl", "rb"))

# Load class_names
class_names = pickle.load(open("class_names.pkl", "rb"))

# Function to predict sentiment
def predict_sentiment(text):
    text = mng.get_clean(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=max_len)
    prediction = model.predict(text)
    return prediction

# Streamlit app
def main():
    st.title("News Classification")
    user_input = st.text_input('Enter your text here:')

    if st.button("Predict"):
        if user_input:
            prediction = predict_sentiment(user_input)
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class]
            st.write('Predicted class:', predicted_label)
        else:
            st.write('Please enter some text to predict.')

if __name__ == "__main__":
    main()
