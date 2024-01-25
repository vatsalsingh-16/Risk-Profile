# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

def preprocess_tweet(tweet):
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    stemmed_content = re.sub('[^a-zA-Z]', ' ', tweet)
    stemmed_content = stemmed_content.lower()

    # Tokenize and apply stemming, excluding stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content.split() if word not in stop_words]

    # Join the stemmed words back into a string
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# Load the dataset
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('./data/training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')
twitter_data.replace({'target': {4: 1}}, inplace=True)

# Preprocess the data
twitter_data['stemmed_content'] = twitter_data['text'].apply(preprocess_tweet)

# Split the data
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'model/model.joblib')
joblib.dump(vectorizer, 'model/vectorizer.joblib')