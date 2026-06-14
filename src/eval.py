# eval.py

import json
import wandb

from sklearn.metrics import classification_report


def evaluate_model(trainer, test_dataset, id2label):

    results = trainer.evaluate()

    preds = trainer.predict(test_dataset).predictions.argmax(-1)

    labels = [
        item["labels"].item()
        for item in test_dataset
    ]

    report = classification_report(
        labels,
        preds,
        target_names=list(id2label.values()),
        output_dict=True
    )

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact(
        "eval-report",
        type="evaluation"
    )

    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)

    return results
