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
git clone https://github.com/g25ait2025/ml_ops_group27_assignmrnt.git

cd iml_ops_group27_assignmrnt
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

https://www.kaggle.com/code/bhoopendrakumarg25/ml-ops-g27

---

# Hugging Face Model

Public Model Repository:

https://huggingface.co/Bhoop-g25ait2025/distilbert-goodreads-genres_g27

---

# Docker Image

Docker Hub Repository:

https://hub.docker.com/repository/docker/g25ait2025/goodreads-classifier_27/general

---

# Weights & Biases Dashboard

Public W&B Project:

https://wandb.ai/g25ait2025-prom-iit-rajasthan/mlops-group27-assignment?nw=nwuserg25ait2025

---

# Team Members

| Name             |                                    Responsibility                                        |      
|------------------|------------------------------------------------------------------------------------------|
| Bhoopendra Kumar | Project lead & coordination across all tasks; repository setup & branch protection;      |
|  (G25AIT2025)    |  Kaggle GPU training of both versions (v1/v2); W&B experiment tracking and integration;  |
|                  |  pushing the best model to Hugging Face Hub                                              |     
| Khushi Bawistale | Dataset download & inspection; reusable data-cleaning / normalisation script;            |
|  (G25AIT2052)    | class-balancing and id2label.json mapping; exploratory data analysis                     |
| Tejaswini        |Dockerfile design and inference.py;local build & end-to-end testing; publishing the public| 
|  (G25AIT2055)    |image to Docker Hub; inference output formatting                                          |
| Kapil Sharma     | CI and Inference GitHub Actions workflows; GitHub & Kaggle secrets configuration; README,|
|  (G25AIT2047)    | W&B comparison write-up and final report                                                 |
---------------------------------------------------------------------------------------------------------------

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
