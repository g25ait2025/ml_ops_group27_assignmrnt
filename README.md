# IITJ MLOps End-to-End Pipeline

An end-to-end MLOps project developed as part of the IIT Jodhpur PGD AI program. This project demonstrates the complete machine learning lifecycle, including data preprocessing, model training, experiment tracking, model deployment, containerization, and CI/CD automation.

## Project Overview

This project uses a Hugging Face Transformer model for text classification and integrates:

- Hugging Face Transformers
- Weights & Biases (W&B)
- Kaggle GPU Training
- Docker
- GitHub Actions
- Hugging Face Model Hub

---

## Repository Structure

```text
iitj-mlops-end-to-end-pipeline/
│
├── src/
│   ├── data.py
│   ├── train.py
│   ├── eval.py
│   ├── inference.py
│   └── utils.py
│
├── notebooks/
│
├── data/
│
├── reports/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── inference.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
├── LICENSE
└── id2label.json
```

---

# Prerequisites

- Python 3.11+
- Git
- Docker
- Hugging Face Account
- Weights & Biases Account

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-org>/iitj-mlops-end-to-end-pipeline.git

cd iitj-mlops-end-to-end-pipeline
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Linux / Mac

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Run data preprocessing:

```bash
python src/data.py
```

This script:

- Loads the dataset
- Removes null values
- Removes duplicates
- Performs preprocessing
- Generates `id2label.json`

---

# Model Training

Run model training:

```bash
python src/train.py
```

Training includes:

- DistilBERT fine-tuning
- Validation
- W&B experiment tracking
- Model checkpointing

---

# Model Evaluation

Run evaluation:

```bash
python src/eval.py
```

Metrics:

- Accuracy
- F1 Score
- Validation Loss

---

# Local Inference

Run prediction:

```bash
python src/inference.py
```

Example:

```python
text = "This book is amazing and full of adventure."
```

Output:

```text
Fantasy
```

---

# Docker Setup

Build Docker image:

```bash
docker build \
-t mlops-inference .
```

Run container:

```bash
docker run \
-e INPUT_TEXT="This book is amazing" \
mlops-inference
```

---

# GitHub Actions

The repository contains two GitHub Actions workflows:

### CI Workflow

Runs automatically on:

- Push to develop
- Pull request to main

Checks:

- Code quality
- Dependency installation
- Linting

### Inference Workflow

Triggered manually using:

```text
Actions
→ Inference
→ Run Workflow
```

Allows users to submit text and receive predictions.

---

# Kaggle Notebooks

### Experiment 1

Kaggle Notebook Link:

<PASTE_KAGGLE_NOTEBOOK_1_URL>

### Experiment 2

Kaggle Notebook Link:

<PASTE_KAGGLE_NOTEBOOK_2_URL>

---

# Hugging Face Model

Public Model Repository:

<PASTE_HUGGINGFACE_MODEL_URL>

Example:

https://huggingface.co/your-team/distilbert-goodreads

---

# Docker Image

Docker Hub Repository:

<PASTE_DOCKERHUB_URL>

Example:

https://hub.docker.com/r/your-team/mlops-inference

---

# Weights & Biases Dashboard

Public W&B Project:

<PASTE_WANDB_URL>

Example:

https://wandb.ai/your-team/mlops-project

---

# Team Members

| Name | Responsibility |
|--------|---------------|
| Member 1 | Repository Setup & GitHub Actions |
| Member 2 | Dataset Preparation |
| Member 3 | Model Training & W&B |
| Member 4 | Docker & Deployment |

---

# Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- Weights & Biases
- Docker
- GitHub Actions
- Kaggle

---

# License

This project is licensed under the MIT License.
