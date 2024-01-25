# sentiment_model.py
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

model = joblib.load('model/model.joblib')
vectorizer = joblib.load('model/vectorizer.joblib')

stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()



def preprocess_tweet(tweet):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', tweet)
    stemmed_content = stemmed_content.lower()

    # Tokenize and apply stemming, excluding stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content.split() if word not in stop_words]

    # Join the stemmed words back into a string
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

def predict_sentiment(tweet):
    processed_tweet = preprocess_tweet(tweet)
    vectorized_tweet = vectorizer.transform([processed_tweet])
    prediction = model.predict(vectorized_tweet)
    return prediction[0]

# Note: Add any additional functions or logic as needed.