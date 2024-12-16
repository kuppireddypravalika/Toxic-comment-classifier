from flask import Flask, request, jsonify
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk

# Initialize Flask app
app = Flask(__name__)

# Load nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load pre-trained TF-IDF vectorizer and model
with open('./artifacts/tfidf_model.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('./artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Offensive words list (example)
OFFENSIVE_WORDS = {"hate", "stupid", "idiot", "fool", "ugly"}

# Initialize helper tools
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    # Remove punctuation and newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove all characters except letters, numbers, and whitespace
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with a space
    return text.strip()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = clean_text(text)  # Apply the cleaning step
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Feature extraction functions
def lexical_features(text):
    tokens = preprocess_text(text)
    return {
        "num_characters": len(text),
        "num_words": len(tokens),
        "num_sentences": len(sent_tokenize(text)),
        "avg_word_length": sum(len(word) for word in tokens) / len(tokens) if tokens else 0
    }

def sentiment(text):
    sentiment_score = sia.polarity_scores(text)['compound']
    return {
        "Sentiment": 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0
    }

def syntactic_features(text):
    tokens = preprocess_text(text)
    pos_counts = Counter(tag for _, tag in pos_tag(tokens))
    return {
        "num_nouns": pos_counts.get('NN', 0),
        "num_verbs": pos_counts.get('VB', 0),
        "num_adjectives": pos_counts.get('JJ', 0),
    }

def domain_specific_features(text):
    tokens = preprocess_text(text)
    all_caps_words = [word for word in tokens if word.isupper()]
    toxic_keywords = [word for word in tokens if word in OFFENSIVE_WORDS]
    return {
        "num_all_caps_words": len(all_caps_words),
        "num_toxic_keywords": len(toxic_keywords)
    }

# Combine all features
def extract_features(text):
    features = {}
    features.update(lexical_features(text))
    features.update(syntactic_features(text))
    features.update(domain_specific_features(text))
    features.update(sentiment(text))
    return features

# Flask endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Step 1: Clean the text
        cleaned_text = clean_text(text)

        # Step 2: Extract features
        feature_dict = extract_features(cleaned_text)

        # Step 3: Convert features to array
        feature_values = np.asarray(list(feature_dict.values()), dtype=float).reshape(1, -1)

        # Step 4: Transform text with TF-IDF
        tfidf_features = tfidf_vectorizer.transform([cleaned_text]).toarray()

        # Step 5: Combine features and TF-IDF vectors
        combined_features = np.hstack((feature_values, tfidf_features))
        print("final dim",combined_features.shape)

        # Step 6: Predict using the loaded model
        prediction = model.predict(combined_features)[0]
        out=""
        if int(prediction)==1:
            out="Toxic"
        else:
            out="Non-Toxic"

        return jsonify({"prediction": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
