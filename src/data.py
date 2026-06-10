import json

print("Dataset cleaning script")

id2label = {
    "0": "negative",
    "1": "positive"
}

with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=4)

print("id2label.json created")
