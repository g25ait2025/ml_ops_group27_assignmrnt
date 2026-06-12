# data.py
# loads the dataset

import gzip
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import clean_text


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []

    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for i, line in enumerate(file):
            reviews.append(json.loads(line)["review_text"])

            if head and i >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def prepare_dataframe(genre_reviews_dict):
    data = []

    for genre, reviews in genre_reviews_dict.items():
        for review in reviews:
            data.append([review, genre])

    df = pd.DataFrame(data, columns=["review_text", "genre"])

    df = df.dropna(subset=["review_text"])

    df["review_text"] = df["review_text"].apply(clean_text)

    df = df[df["review_text"] != ""]
    df = df.drop_duplicates(subset=["review_text"])

    return df


def split_data(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["genre"]
    )

    return train_df, test_df
