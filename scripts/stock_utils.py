# scripts/stock_utils.py

import yfinance as yf
import pandas as pd


def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df


def calculate_sma(df, window=14):
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df


def calculate_ema(df, window=14):
    df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df


def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, short=12, long=26, signal=9):
    short_ema = df['Close'].ewm(span=short, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long, adjust=False).mean()

    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df
