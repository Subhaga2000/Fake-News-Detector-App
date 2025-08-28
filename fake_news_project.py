import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Machine Learning Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

# --- Step 1.1: Load and combine the datasets ---
print("Loading datasets...")
try:
    # Using the provided full paths from your traceback
    fake_df = pd.read_csv('C:/Users/ASUS/Documents/fake_news_detector/data/Fake.csv')
    real_df = pd.read_csv('C:/Users/ASUS/Documents/fake_news_detector/data/True.csv')
except FileNotFoundError:
    print("Error: Dataset files not found. Make sure 'Fake.csv' and 'True.csv' are in the 'data' folder.")
    exit()

# --- Step 1.2: Add a label column to each DataFrame ---
fake_df['label'] = 0  # 0 for fake news
real_df['label'] = 1  # 1 for real news

# --- Step 1.3: Combine them into a single DataFrame ---
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.dropna()
print("Datasets loaded, combined, and missing values handled.")

# --- Step 1.4: Resample the dataset to balance the classes (FIXED LOGIC) ---
print("Balancing the dataset...")
# Separate classes
df_class_0 = df[df.label == 0] # Fake news
df_class_1 = df[df.label == 1] # Real news

# Determine which class is majority and which is minority
if len(df_class_1) > len(df_class_0):
    df_majority = df_class_1
    df_minority = df_class_0
else:
    df_majority = df_class_0
    df_minority = df_class_1

# Undersample the majority class to match the minority class size
df_majority_undersampled = resample(df_majority,
                                    replace=False,    # sample without replacement
                                    n_samples=len(df_minority),  # to match minority class size
                                    random_state=42)  # for reproducibility

# Combine minority class with the undersampled majority class
df_balanced = pd.concat([df_majority_undersampled, df_minority])
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True) # Shuffle the balanced dataframe
print(f"Dataset is now balanced. New shape: {df_balanced.shape}")
print(df_balanced['label'].value_counts())

# --- Step 2.1: Initialize stemmer and stopwords ---
print("\nStarting text preprocessing...")
ps = PorterStemmer()
try:
    stopwords_english = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    import nltk
    nltk.download('stopwords')
    stopwords_english = set(stopwords.words('english'))

# --- Step 2.2: Define the preprocessing function ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters and replace with space
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords_english]
    return ' '.join(words)

# --- Step 2.3: Combine and apply preprocessing ---
df_balanced['combined_text'] = df_balanced['title'].fillna('') + ' ' + df_balanced['text'].fillna('')
df_balanced['clean_text'] = df_balanced['combined_text'].apply(preprocess_text)
print("Text preprocessing complete.")

# --- Step 3.1: Split data into training and testing sets ---
X = df_balanced['clean_text']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3.2: Tokenize and pad sequences ---
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Save the tokenizer for the Streamlit app
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# --- Step 3.3: Build the LSTM model ---
print("\nBuilding and training the model...")
model = Sequential()
model.add(Embedding(max_words, 128))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# --- Step 3.4: Compile and train the model ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

# --- Step 3.5: Evaluate the model ---
print("\nEvaluating the model on the test set...")
y_pred_prob = model.predict(X_test_pad)
y_pred_class = (y_pred_prob > 0.5).astype("int32")
print(classification_report(y_test, y_pred_class, target_names=['Fake', 'Real']))

# --- Step 3.6: Save the trained model ---
model.save('fake_news_model.h5')
print("\nModel training and saving complete. You can now run the Streamlit app.")