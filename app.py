from flask import Flask, render_template, request
from model.sentiment_model import predict_sentiment
from flask import Flask, render_template, request, redirect, url_for
from model.sentiment_model import predict_sentiment
from io import BytesIO
import base64

app = Flask(__name__, template_folder='app/templates')

# List to store entered tweets
all_tweets = []

from app import routes

def count_sentiments():
    total_positive = sum(1 for tweet_entry in all_tweets if tweet_entry['result'] == 1)
    total_negative = sum(1 for tweet_entry in all_tweets if tweet_entry['result'] == 0)
    return total_positive, total_negative

@app.route('/')
def index():
    total_positive, total_negative = count_sentiments()
    return render_template('index.html', all_tweets=all_tweets, total_positive=total_positive, total_negative=total_negative)

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    result = predict_sentiment(tweet)

    # Update the list of tweets with the sentiment result
    all_tweets.append({'tweet': tweet, 'result': result})

    # Count positive and negative tweets
    total_positive, total_negative = count_sentiments()

    # Calculate risk percentage
    total_tweets = len(all_tweets)
    risk_percentage = (total_negative / total_tweets) * 100 if total_tweets > 0 else 0

    return render_template('index.html', tweet=tweet, result=result, all_tweets=all_tweets,
                           total_positive=total_positive, total_negative=total_negative, risk_percentage=risk_percentage)

def count_sentiments():
    total_positive = sum(1 for tweet_entry in all_tweets if tweet_entry['result'] == 1)
    total_negative = sum(1 for tweet_entry in all_tweets if tweet_entry['result'] == 0)
    return total_positive, total_negative






if __name__ == '__main__':
    app.run(debug=True)
