# train.py

import wandb

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from utils import compute_metrics


MODEL_NAME = "distilbert-base-uncased"


def train_model(train_dataset, test_dataset, id2label, label2id):

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer, model, tokenizer
