# scripts/correlation_utils.py

import pandas as pd


def compute_daily_returns(df):
    df['Daily Return'] = df['Close'].pct_change()
    df['Date'] = df.index.date
    return df


def merge_sentiment_and_returns(sentiment_df, returns_df):
    merged = pd.merge(sentiment_df, returns_df[['Date', 'Daily Return']], on='Date', how='inner')
    return merged


def calculate_correlation(merged_df):
    return merged_df['Sentiment'].corr(merged_df['Daily Return'])
