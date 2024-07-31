import streamlit as st
from googletrans import Translator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import joblib

# Initialize the Translator
translator = Translator()

# Load your pre-trained model
model_form = tf.keras.models.load_model('D:/itconvert/text-detection/nlp-streamlit/best_model.h5')

# Define your tokenizer (replace this with your tokenizer instance)


# Save the model
token_form = joblib.load("D:/itconvert/text-detection/nlp-streamlit/tokenizer.pkl")
# Streamlit UI
st.title("Text Sentiment Analysis")
st.write("Enter text to analyze whether it's a potential suicide post or not.")

# Text input from user
user_text = st.text_area("Text to Analyze", "")

if st.button('Analyze'):
    if user_text:
        # Translate text from Indonesian to English
        translated = translator.translate(user_text, src='id', dest='en')
        translated_text = translated.text

        # st.write("Translated Text:")
        # st.write(translated_text)

        # Prepare text for prediction
        twt = [translated_text]
        twt = token_form.texts_to_sequences(twt)
        twt = pad_sequences(twt, maxlen=50)

        # Predict
        prediction = model_form.predict(twt)[0][0]
        
        if prediction > 0.5:
            st.write("Prediction: Potensi bunuh diri")
            st.write(f"Akurasi: {prediction*100:.2f}%")
        else:
            st.write("Prediction: Bukan Bunuh diri")
            st.write(f"Akurasi: {prediction*100:.2f}%")

    else:
        st.write("Please enter text for analysis.")
