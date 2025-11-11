import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TITLE
st.title("Financial News Sentiment Classifier")
st.write(
    "Enter a financial news headline below. The model will predict whether the sentiment is **positive**, **negative**, or **neutral**."
)

# LOAD OBJECTS
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
model = load_model('sentiment_model.h5')

# USER INPUT
headline = st.text_input("Headline:", "")

if st.button("Predict sentiment"):
    # Text preprocessing
    seq = tokenizer.texts_to_sequences([headline])
    pad = pad_sequences(seq, maxlen=30)  # match length used during training
    pred = model.predict(pad)
    label = encoder.inverse_transform([np.argmax(pred)])
    st.markdown(f"**Predicted sentiment:** `{label[0]}`")
