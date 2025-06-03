# scripts/news_sentiment.py

import pandas as pd
from textblob import TextBlob


def load_news_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df


def compute_sentiment(df):
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity

    df['Sentiment'] = df['Headline'].astype(str).apply(get_sentiment)
    return df


def daily_avg_sentiment(df):
    return df.groupby('Date')['Sentiment'].mean().reset_index()
