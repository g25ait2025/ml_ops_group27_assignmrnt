# utils.py

import re
import string

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }
