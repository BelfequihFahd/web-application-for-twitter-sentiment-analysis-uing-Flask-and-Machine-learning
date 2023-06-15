from flask import Flask, render_template, request
import joblib
import re
import pickle
import time
import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model_extra.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


def predict_sentiment(text):
    # Preprocess the input text
    text_vectorized = vectorizer.transform([text])

    # Make predictions using the loaded model
    sentiment = model.predict(text_vectorized)
    mapping = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Irrelevant"}

    return mapping[sentiment[0]]


def get_tweets(username):
    query = 'from:' + username + ' since:2023-06-12 until:2023-06-12'
    scraper = sntwitter.TwitterSearchScraper(query)
    tweet_data = []
    for i, tweet in enumerate(scraper.get_items()):
        tweet_list = [
            tweet.content
        ]
        tweet_data.append(tweet_list)
        if i == 9:
            break
    df = pd.DataFrame(tweet_data, columns=['content'])
    predicted_values = []

    df['sentiment'] = df['content'].apply(predict_sentiment)

    return df


def preprocess_text(text):
    # Clean the text by removing special characters and converting to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    username = request.form.get('username', '')

    if text != '':
        data = {
            'content': [text],
            'sentiment': [predict_sentiment(text)]
        }
        print(data)
        df = pd.DataFrame(data)

    elif username != '':
        df = get_tweets(username)

    else:
        df = None
    return render_template('index.html', data=df)


if __name__ == '__main__':
    app.run()
