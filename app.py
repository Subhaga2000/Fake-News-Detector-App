import streamlit as st
import tensorflow as tf
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Set page config first ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️")

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_resources():
    try:
        # Load the model. Ensure the path is correct if not in the same directory.
        model = tf.keras.models.load_model('fake_news_model.h5')
        # Load the tokenizer. Ensure the path is correct if not in the same directory.
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer. Make sure to run 'main_script.py' first to generate these files. Details: {e}")
        return None, None

model, tokenizer = load_resources()

# --- Preprocessing Function ---
ps = PorterStemmer()
try:
    stopwords_english = set(stopwords.words('english'))
except LookupError:
    # This block is for Streamlit Cloud deployment where NLTK data may not be pre-installed
    import nltk
    nltk.download('stopwords')
    stopwords_english = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords_english]
    return ' '.join(words)

# --- Streamlit UI and Prediction Logic ---
st.title("🕵️ Fake News Detector")
st.markdown("Enter a news article or headline below to determine if it's likely fake or real.")

if model and tokenizer:
    input_text = st.text_area("Enter news text:", height=250, placeholder="Paste your article here...")

    if st.button("Analyze News", use_container_width=True, type="primary"):
        if input_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                cleaned_text = preprocess_text(input_text)
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                padded_sequences = pad_sequences(sequences, maxlen=200)
                
                prediction_prob = model.predict(padded_sequences)[0][0]
                
                st.subheader("Analysis Result")
                if prediction_prob > 0.5:
                    st.error(f"This news is likely **FAKE** 🤥")
                    st.write(f"Confidence Score: **{prediction_prob:.2f}**")
                else:
                    st.success(f"This news is likely **REAL** ✅")
                    st.write(f"Confidence Score: **{1 - prediction_prob:.2f}**")
else:
    st.info("The application is not ready. Please run the training script first to generate the model and tokenizer files.")