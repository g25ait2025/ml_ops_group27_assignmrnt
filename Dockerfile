FROM python:3.11-slim

ARG HF_MODEL_NAME=Bhoop-g25ait2025/distilbert-goodreads-genres_g27

ENV HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]