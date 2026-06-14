# inference.py

import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

MODEL_NAME = os.getenv(
    "HF_MODEL_NAME",
    "Bhoop-g25ait2025/distilbert-goodreads-genres_g27"
)

INPUT_TEXT = os.getenv(
    "INPUT_TEXT",
    "This book was amazing and full of magic."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

result = classifier(INPUT_TEXT)

print(result)
