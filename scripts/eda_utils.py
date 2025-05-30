# scripts/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def load_dataset(path):
    return pd.read_csv(path)


def plot_articles_per_day(df):
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    daily_counts = df.groupby('date_only').size()
    daily_counts.plot(kind='line', title='Articles per Day', figsize=(10, 4))
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def topic_modeling(texts, n_topics=5, n_words=10):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts.fillna(''))

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic in lda.components_:
        keywords = [feature_names[i] for i in topic.argsort()[-n_words:]]
        topics.append(keywords)

    return topics


def plot_articles_by_hour(df):
    df['hour'] = df['date'].dt.hour
    df['hour'].value_counts().sort_index().plot(kind='bar', title='Articles by Hour (UTC-4)', figsize=(8, 4))
    plt.xlabel('Hour')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
