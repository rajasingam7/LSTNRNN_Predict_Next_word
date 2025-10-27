import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import os

st.title("LSTM RNN Next Word Predictor")

# Paths to model and tokenizer
MODEL_PATH = 'lstm_rnn_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

# Load model safely
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.warning("Model file not found!")
    model = None

# Load tokenizer safely
if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        st.success("Tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        tokenizer = None
else:
    st.warning("Tokenizer file not found!")
    tokenizer = None

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not model or not tokenizer:
        return "Model or tokenizer not loaded!"
    
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    try:
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
    except Exception as e:
        return f"Prediction error: {e}"
    
    return None

# Streamlit app input
input_text = st.text_input("Enter a sequence of words:")

if st.button("Predict Next Word"):
    if model and tokenizer:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Predicted Next Word: {next_word}")
    else:
        st.error("Cannot predict because model or tokenizer is not loaded.")
