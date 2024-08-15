import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app configuration
st.set_page_config(page_title="Next Word Prediction", page_icon="🔮")

# Custom styling for dark mode
st.markdown("""
    <style>
        .stTextInput>div>div>textarea {
            border: 2px solid #1e1e1e;
            border-radius: 8px;
            background-color: #333;
            color: #ddd;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .info-section {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            color: #ddd;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #1e1e1e;
            border-top: 1px solid #333;
            color: #ddd;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("Next Word Prediction With LSTM and GRU 🔮")

# Information about the project
st.markdown("""
    <div class="info-section">
        <h3>About This Project</h3>
        <p>
            This application predicts the next word in a sequence using an LSTM model. The model is trained on a large corpus of text data and is capable of generating contextually relevant predictions based on the provided input sequence.
            <br><br>
            <strong>Why LSTM?</strong> 
            Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) that are well-suited for sequence prediction tasks. They are capable of learning long-term dependencies in text sequences, making them ideal for predicting the next word in a sequence.
            <br><br>
            <strong>Future Work:</strong> 
            In future updates, I plan to explore and implement more advanced architectures, such as GRUs and Transformer-based models, to enhance prediction accuracy and performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

# User input
input_text = st.text_input("Enter the sequence of words", "Enter Text")

if st.button("Predict Next Word"):
    if input_text:
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'**Next word:** {next_word}')
    else:
        st.error('Please enter a sequence of words to predict the next word.')

# Footer with styling
st.markdown("""
    <div class="footer">
        Made with ❤️ using Streamlit and TensorFlow.
    </div>
    """, unsafe_allow_html=True)
