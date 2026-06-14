# %% [markdown] {"id":"URKn8K0HydhV","jupyter":{"outputs_hidden":false}}
# # Training and Fine-Tuning BERT for Classification
# ## Classfying Goodreads Reviews By Book Genre
# 
# By Maria Antoniak, Melanie Walsh, and the [AI for Humanists](https://aiforhumanists.com/) Team
# 
# Updated: 2024-11-05
# <br></br>
# 
# This notebook will demonstrate how users can train and fine-tune a BERT model for classification with the popular HuggingFace `transformers` Python library.
# 
# We will fine-tune a BERT model on Goodreads reviews from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) with the goal of predicting the genre of the book being reviewed. The genres include:
# - poetry
# - comics & graphic
# - fantasy & paranormal
# - history & biography
# - mystery, thriller, & crime
# - romance
# - young adult  
# 
# **Basic steps involved in using BERT and HuggingFace:**
# 1. Divide your data into training and test sets.
# 2. Encode your data into a format BERT will understand.
# 3. Combine your data and labels into datset objects.
# 4. Load the pre-trained BERT model.
# 5. Fine-tune the model using your training data.
# 6. Predict new labels and evaluate performance on your test data.

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T09:21:52.525755Z","iopub.execute_input":"2026-06-11T09:21:52.526612Z","iopub.status.idle":"2026-06-11T09:21:52.620008Z","shell.execute_reply.started":"2026-06-11T09:21:52.526568Z","shell.execute_reply":"2026-06-11T09:21:52.619202Z"},"jupyter":{"outputs_hidden":false}}
# In your Kaggle Notebook — run this first
from kaggle_secrets import UserSecretsClient
 
secrets = UserSecretsClient()
WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
HF_TOKEN  	= secrets.get_secret('HF_TOKEN')
 
import os
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['HF_TOKEN']      = HF_TOKEN

# %% [markdown] {"id":"y73xWUkpU1AC","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Import necessary Python libraries and modules**

# %% [markdown] {"id":"BJ9kmfrO3M_w","jupyter":{"outputs_hidden":false}}
# First, we will import necessary Python libraries and modules. These include as `gdown`, for downloading large files from Google Drive (where we will get our UCSD Goodreads reviews), as well as scikit-learn (`sklearn`) and PyTorch (`torch`), for various machine learning tools.

# %% [code] {"id":"BNxmMnzoccfm","outputId":"65e092c0-7a80-4801-d42c-2c07ed4fe259","execution":{"iopub.status.busy":"2026-06-11T09:21:52.621287Z","iopub.execute_input":"2026-06-11T09:21:52.621659Z","iopub.status.idle":"2026-06-11T09:21:56.119726Z","shell.execute_reply.started":"2026-06-11T09:21:52.621602Z","shell.execute_reply":"2026-06-11T09:21:56.119000Z"},"jupyter":{"outputs_hidden":false}}
!pip3 install -U transformers

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T09:21:56.121240Z","iopub.execute_input":"2026-06-11T09:21:56.121466Z","iopub.status.idle":"2026-06-11T09:21:59.583507Z","shell.execute_reply.started":"2026-06-11T09:21:56.121439Z","shell.execute_reply":"2026-06-11T09:21:59.582736Z"},"jupyter":{"outputs_hidden":false}}
!pip install --upgrade huggingface_hub

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T09:21:59.586178Z","iopub.execute_input":"2026-06-11T09:21:59.586416Z","iopub.status.idle":"2026-06-11T09:22:04.478267Z","shell.execute_reply.started":"2026-06-11T09:21:59.586390Z","shell.execute_reply":"2026-06-11T09:22:04.477233Z"},"jupyter":{"outputs_hidden":false}}
!pip uninstall -y huggingface_hub
!pip install huggingface_hub

# %% [code] {"id":"x9Si6kIWcULv","execution":{"iopub.status.busy":"2026-06-11T09:22:04.479920Z","iopub.execute_input":"2026-06-11T09:22:04.480302Z","iopub.status.idle":"2026-06-11T09:22:04.490526Z","shell.execute_reply.started":"2026-06-11T09:22:04.480258Z","shell.execute_reply":"2026-06-11T09:22:04.489718Z"},"jupyter":{"outputs_hidden":false}}
# Basic Python modules
from collections import defaultdict
import random
import pickle

# For downloading large files from Google Drive
# https://github.com/wkentaro/gdown
import gdown

# For working with gzip files
# https://docs.python.org/3/library/gzip.html
import gzip

# For working with JSON files
import json

# For data manipulation and analysis
import pandas as pd
import numpy as np

# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# For deep learning
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import torch

# For plotting and data visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
sns.set(style='ticks', font_scale=1.2)

# %% [markdown] {"id":"4N0RIalt28yL","jupyter":{"outputs_hidden":false}}
# The HuggingFace [`transformers` Python library](https://huggingface.co/transformers/installation.html) is included in Colab by default now, so we do not need to install it (but this is how you would install it with `pip`).

# %% [markdown] {"id":"f29CPpKi3__Q","jupyter":{"outputs_hidden":false}}
# From `transformers`, we will import modules for `DistilBert`, a *distilled* or smaller version of a BERT model that runs more quickly and uses less computing power. This makes it ideal for those just getting started with BERT.

# %% [code] {"id":"8ARMrKfmceAb","execution":{"iopub.status.busy":"2026-06-11T09:22:04.491767Z","iopub.execute_input":"2026-06-11T09:22:04.492087Z","iopub.status.idle":"2026-06-11T09:22:04.502641Z","shell.execute_reply.started":"2026-06-11T09:22:04.492052Z","shell.execute_reply":"2026-06-11T09:22:04.501740Z"},"jupyter":{"outputs_hidden":false}}
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# %% [markdown] {"id":"Q70KYS_NVSDL","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Set parameters and file paths**

# %% [code] {"id":"piWsW9ZeaP_D","execution":{"iopub.status.busy":"2026-06-11T09:22:04.503613Z","iopub.execute_input":"2026-06-11T09:22:04.504301Z","iopub.status.idle":"2026-06-11T09:22:04.515059Z","shell.execute_reply.started":"2026-06-11T09:22:04.504276Z","shell.execute_reply":"2026-06-11T09:22:04.514382Z"},"jupyter":{"outputs_hidden":false}}
# This is the name of the BERT model that we want to use.
# We're using DistilBERT to save space (it's a distilled version of the full BERT model),
# and we're going to use the cased (vs uncased) version.
model_name = 'distilbert-base-cased'

# This is the name of the program management system for NVIDIA GPUs. We're going to send our code here.
device_name = 'cuda'

# This is the maximum number of tokens in any document sent to BERT.
max_length = 512

# This is the name of the directory where we'll save our model. You can name it whatever you want.
cached_model_directory_name = 'distilbert-reviews-genres_g27'

# %% [markdown] {"id":"tUjeaqiTehPY","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Load and sample Goodreads data**

# %% [markdown] {"id":"SaYqqj7N8aRs","jupyter":{"outputs_hidden":false}}
# In this cell, we create a Python dictionary with each genre and the link to the corresponding UCSD Goodreads review data for that genre.
# 
# *If you manually click on any of the URLs, you will be able to download the data for that genre. For example, here's the link for poetry: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz*

# %% [code] {"id":"J7pcelDZdrcC","execution":{"iopub.status.busy":"2026-06-11T09:22:04.516152Z","iopub.execute_input":"2026-06-11T09:22:04.516515Z","iopub.status.idle":"2026-06-11T09:22:04.526867Z","shell.execute_reply.started":"2026-06-11T09:22:04.516491Z","shell.execute_reply":"2026-06-11T09:22:04.526166Z"},"jupyter":{"outputs_hidden":false}}
# This is where our target data is hosted on the web. You only need these paths for the book review dataset.

# Source: https://mengtingwan.github.io/data/goodreads.html#datasets

genre_url_dict = {'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
                  'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
                  'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
                  'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
                  'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
                  'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
                  'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
                  'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'}

# %% [markdown] {"id":"UZC5SadPAZEc","jupyter":{"outputs_hidden":false}}
# Next we loop through this dictionary and use `gdown` to download the Goodreads review data for each genre from Google Drive.

# %% [markdown] {"id":"WUxZo1Lz6agB","jupyter":{"outputs_hidden":false}}
# Now we will load the first 100,000 reviews from each link and randomly sample 2,000 reviews.

# %% [code] {"id":"H7gXZV4V4NZ0","outputId":"42b89d8e-e242-4de2-b571-768810e060e8","execution":{"iopub.status.busy":"2026-06-11T09:22:04.528035Z","iopub.execute_input":"2026-06-11T09:22:04.528269Z","iopub.status.idle":"2026-06-11T09:22:11.283896Z","shell.execute_reply.started":"2026-06-11T09:22:04.528229Z","shell.execute_reply":"2026-06-11T09:22:11.282887Z"},"jupyter":{"outputs_hidden":false}}
import requests
# Stream reviews from URL and collect a subset
def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    print(response)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1

            # Stop if we have reached the 100,000 limit
            if head is not None and count >= head:
                break

    # Return random sample of reviews
    return random.sample(reviews, min(sample_size, len(reviews)))

# Reviews by genre
genre_reviews_dict = {}

# Load reviews for each genre
for genre, url in genre_url_dict.items():
    print(f'Loading reviews for genre: {genre}')
    genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=2000)

# %% [code] {"execution":{"iopub.status.busy":"2026-06-11T09:22:11.287100Z","iopub.execute_input":"2026-06-11T09:22:11.287446Z","iopub.status.idle":"2026-06-11T09:22:12.200770Z","shell.execute_reply.started":"2026-06-11T09:22:11.287421Z","shell.execute_reply":"2026-06-11T09:22:12.199891Z"}}
#  Bhoopendra Kumar,11-06-26,Code add to data cleaning and normalization

# Create DataFrame from loaded reviews

data = []

for genre, reviews in genre_reviews_dict.items():
    for review in reviews:
        data.append([review, genre])

df = pd.DataFrame(data, columns=["review_text", "genre"])

# ==========================
# Data Cleaning
# ==========================

import re
import string

# Remove missing values
df = df.dropna(subset=["review_text"])

# Lowercase + punctuation removal
def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["review_text"] = df["review_text"].apply(clean_text)

# Remove empty reviews
df = df[df["review_text"] != ""]

# Remove duplicates
df = df.drop_duplicates(subset=["review_text"])

print("Final Shape:", df.shape)

# %% [markdown] {"id":"5hUYAM95IVyB","jupyter":{"outputs_hidden":false}}
# Let's preview a couple of the key-value pairs in `genre_reviews_dict`

# %% [code] {"id":"Rf5RDXqGFQJy","outputId":"3dab2970-d7d0-440e-d5c7-f7ffd7f07a01","execution":{"iopub.status.busy":"2026-06-11T09:22:12.201818Z","iopub.execute_input":"2026-06-11T09:22:12.202159Z","iopub.status.idle":"2026-06-11T09:22:12.207536Z","shell.execute_reply.started":"2026-06-11T09:22:12.202122Z","shell.execute_reply":"2026-06-11T09:22:12.206778Z"},"jupyter":{"outputs_hidden":false}}
 for _genre, _reviews in genre_reviews_dict.items():
    print(_genre)
    print(random.sample(_reviews, 1)[0])

# %% [markdown] {"id":"kJBuJ-7yFc9b","jupyter":{"outputs_hidden":false}}
# Here we use `pickle` to save this Python dictionary to a `.pickle` file so we can easily load it later.
# 
# *The `pickle` module allows you to save and load Python objects like lists and dictionaries.*

# %% [code] {"id":"YnSpBCOuuKu-","execution":{"iopub.status.busy":"2026-06-11T09:22:12.208627Z","iopub.execute_input":"2026-06-11T09:22:12.208899Z","iopub.status.idle":"2026-06-11T09:22:12.241877Z","shell.execute_reply.started":"2026-06-11T09:22:12.208868Z","shell.execute_reply":"2026-06-11T09:22:12.241143Z"},"jupyter":{"outputs_hidden":false}}
pickle.dump(genre_reviews_dict, open('genre_reviews_dict.pickle', 'wb'))
# genre_reviews_dict = pickle.load(open('genre_reviews_dict.pickle', 'rb'))

# %% [markdown] {"id":"v10htL_G-OFs","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Split the data into training and test sets**

# %% [markdown] {"id":"y0qBs3MHGOSI","jupyter":{"outputs_hidden":false}}
# When training a machine learning model, it is necessary to split your training data into two parts: a "training" set and a "test" set.
# 
# We will train our BERT model on the "training" set of Goodreads reviews and then we will evaluate how well it is performing by running it on the "test" set of Goodreads reviews that the model has never seen before.
# 
# Normally, to tune the hyperparameters, you should also create a "validation" set for tuning, and only use the "test" set once, at the end of all tuning. For simplicity, in this tutorial, we will only using a training and test set.

# %% [code] {"id":"h94eg2wlfi6y","execution":{"iopub.status.busy":"2026-06-11T09:22:12.242776Z","iopub.execute_input":"2026-06-11T09:22:12.243090Z","iopub.status.idle":"2026-06-11T09:22:12.265091Z","shell.execute_reply.started":"2026-06-11T09:22:12.243057Z","shell.execute_reply":"2026-06-11T09:22:12.264503Z"},"jupyter":{"outputs_hidden":false}}
#Bhoopendra Kumar,11-06-26,Code commented
#train_texts = []
#train_labels = []

#test_texts = []
#test_labels = []

#for _genre, _reviews in genre_reviews_dict.items():

#  _reviews = random.sample(_reviews, 1000) # Use a very small set as an example.

#  for _review in _reviews[:800]:
#    train_texts.append(_review)
#    train_labels.append(_genre)
#  for _review in _reviews[800:]:
#    test_texts.append(_review)
#    test_labels.append(_genre)

# Bhoopendra Kumar,11-06-26,New code added to resize and structure the data
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["genre"]
)

train_texts = train_df["review_text"].tolist()
test_texts = test_df["review_text"].tolist()

train_labels = train_df["genre"].tolist()
test_labels = test_df["genre"].tolist()

# %% [markdown] {"id":"jJDqqi5tLAqG","jupyter":{"outputs_hidden":false}}
# Show how many Goodreads reviews and labels we have in each category: 6400 training reviews, 6400 training labels (genres), 1600 test reviews, 1600 test labels (genre)

# %% [code] {"id":"4tShbGfG-VMe","outputId":"c4d9a2e5-6ec4-43cf-ee07-790955fcffd9","execution":{"iopub.status.busy":"2026-06-11T09:22:12.266015Z","iopub.execute_input":"2026-06-11T09:22:12.266345Z","iopub.status.idle":"2026-06-11T09:22:12.272289Z","shell.execute_reply.started":"2026-06-11T09:22:12.266320Z","shell.execute_reply":"2026-06-11T09:22:12.271475Z"},"jupyter":{"outputs_hidden":false}}
len(train_texts), len(train_labels), len(test_texts), len(test_labels)

# %% [markdown] {"id":"O-34oep8LNKw","jupyter":{"outputs_hidden":false}}
# Here's an example of a training label and review:

# %% [code] {"id":"dcxvqUsdf5Sm","outputId":"311072bd-c9d2-4231-9608-73ed1834737b","execution":{"iopub.status.busy":"2026-06-11T09:22:12.273271Z","iopub.execute_input":"2026-06-11T09:22:12.273658Z","iopub.status.idle":"2026-06-11T09:22:12.284310Z","shell.execute_reply.started":"2026-06-11T09:22:12.273615Z","shell.execute_reply":"2026-06-11T09:22:12.283588Z"},"jupyter":{"outputs_hidden":false}}
train_labels[0], train_texts[0]

# %% [markdown] {"id":"0FoaXKbKjXRX","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Run a baseline model (logistic regression)**
# 
# Here we train and evaluate a simple TF-IDF baseline model using logistic regression.
# 
# We find better-than-random performance, even for a very small dataset. We'll see whether BERT can beat this good baseline!

# %% [code] {"id":"RBTyTh8Ui2D3","execution":{"iopub.status.busy":"2026-06-11T09:22:12.285263Z","iopub.execute_input":"2026-06-11T09:22:12.285613Z","iopub.status.idle":"2026-06-11T09:22:13.536958Z","shell.execute_reply.started":"2026-06-11T09:22:12.285578Z","shell.execute_reply":"2026-06-11T09:22:13.535992Z"},"jupyter":{"outputs_hidden":false}}
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# %% [markdown] {"id":"gMjmweu6MIQU","jupyter":{"outputs_hidden":false}}
# We train a logistic regression model from scikit-learn on the Goodreads training data, and then we use the trained model to make predictions on our Goodreads review test set.

# %% [code] {"id":"R92S7JZfjiaC","execution":{"iopub.status.busy":"2026-06-11T09:22:13.538086Z","iopub.execute_input":"2026-06-11T09:22:13.538414Z","iopub.status.idle":"2026-06-11T09:22:30.688514Z","shell.execute_reply.started":"2026-06-11T09:22:13.538380Z","shell.execute_reply":"2026-06-11T09:22:30.687725Z"},"jupyter":{"outputs_hidden":false}}
model = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
predictions = model.predict(X_test)

# %% [markdown] {"id":"s6YmW7WZMhh3","jupyter":{"outputs_hidden":false}}
# We can use scikit-learn's `classification_report` function to evaluate how well the logistic regression model's predictions match up with the true labels for the Goodreads reviews.
# 
# Importantly, we can see that our average scores are above random performance (we have 8 classes, so random performance would be ~0.2).

# %% [code] {"id":"UeJd8ogKjpg0","outputId":"203d6d30-f287-49c2-c082-d29a693c7e95","execution":{"iopub.status.busy":"2026-06-11T09:22:30.689440Z","iopub.execute_input":"2026-06-11T09:22:30.689711Z","iopub.status.idle":"2026-06-11T09:22:30.745024Z","shell.execute_reply.started":"2026-06-11T09:22:30.689680Z","shell.execute_reply":"2026-06-11T09:22:30.744408Z"},"jupyter":{"outputs_hidden":false}}
print(classification_report(test_labels, predictions))

# %% [markdown] {"id":"Aow3FPpppZVE","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Encode data for BERT**
# 
# We're going to transform our texts and labels into a format that BERT (via Huggingface and PyTorch) will understand. This is called *encoding* the data.
# 
# Here are the steps we need to follow:
# 
# 1. The labels&mdash;in this case, Goodreads genres&mdash;need to be turned into integers rather than strings.
# 
# 2. The texts&mdash;in this case, Goodreads reviews&mdash;need to be truncated if they're more than 512 tokens or padded if they're fewer than 512 tokens. The tokens, or words in the texts, also need to be separated into "word pieces" and matched to their embedding vectors.
# 
# 3. We need to add special tokens to help BERT:
# 
# | BERT special token | Explanation |
# | --------------| ---------|
# | [CLS] | Start token of every document. |
# | [SEP] | Separator between each sentence |
# | [PAD] | Padding at the end of the document as many times as necessary, up to 512 tokens |
# |  &#35;&#35; | Start of a "word piece" |

# %% [markdown] {"id":"GRs0dEIoUZtV","jupyter":{"outputs_hidden":false}}
# Here we will load `DistilBertTokenizerFast` from the HuggingFace library, which will do all the work of encoding the texts for us. The `tokenizer()` will break word tokens into word pieces, truncate to 512 tokens, and add padding and special BERT tokens.

# %% [code] {"id":"9BEvRqpGVMUD","outputId":"da4bb001-a42d-41b5-bad4-90199302b577","execution":{"iopub.status.busy":"2026-06-11T09:22:30.747446Z","iopub.execute_input":"2026-06-11T09:22:30.747752Z","iopub.status.idle":"2026-06-11T09:22:31.002980Z","shell.execute_reply.started":"2026-06-11T09:22:30.747721Z","shell.execute_reply":"2026-06-11T09:22:31.002253Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) # The model_name needs to match our pre-trained model.
print("Tokenizer loaded successfully")

# %% [markdown] {"id":"oj8X7B30UvSj","jupyter":{"outputs_hidden":false}}
# Here we will create a map of our labels, or Goodreads genres, to integer keys. We take the unique labels, and then we make a dictionary that associates each label/tag with an integer.
# 
# **Note:** HuggingFace documentation sometimes refers to "labels" as "tags" but these are the same thing. We use "labels" throughout this notebook for clarity.

# %% [code] {"id":"tSuo8gktjsVR","execution":{"iopub.status.busy":"2026-06-11T09:22:31.003881Z","iopub.execute_input":"2026-06-11T09:22:31.004287Z","iopub.status.idle":"2026-06-11T09:22:31.010923Z","shell.execute_reply.started":"2026-06-11T09:22:31.004260Z","shell.execute_reply":"2026-06-11T09:22:31.010189Z"},"jupyter":{"outputs_hidden":false}}

#commeted Bhoopendra 11-06-26, to save id2label.json
#unique_labels = set(label for label in train_labels)
#label2id = {label: id for id, label in enumerate(unique_labels)}
#id2label = {id: label for label, id in label2id.items()}

# Create deterministic label mappings
unique_labels = sorted(list(set(train_labels)))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("Label Mapping:")
print(label2id)

# Save id2label.json (required for assignment)
with open("id2label.json", "w") as f:
    json.dump(
        {str(k): v for k, v in id2label.items()},
        f,
        indent=4
    )

print("Saved id2label.json")

# %% [code] {"id":"M_iAWMtBpfhj","outputId":"505cdd65-aafa-4766-c757-bf11c5d92839","execution":{"iopub.status.busy":"2026-06-11T09:22:31.011909Z","iopub.execute_input":"2026-06-11T09:22:31.012204Z","iopub.status.idle":"2026-06-11T09:22:31.028474Z","shell.execute_reply.started":"2026-06-11T09:22:31.012171Z","shell.execute_reply":"2026-06-11T09:22:31.027773Z"},"jupyter":{"outputs_hidden":false}}
label2id.keys()

# %% [code] {"id":"vle8EgkelwRa","outputId":"a6bfa4bd-93c9-4f1a-86bf-9517ce7c9fa6","execution":{"iopub.status.busy":"2026-06-11T09:22:31.029315Z","iopub.execute_input":"2026-06-11T09:22:31.029622Z","iopub.status.idle":"2026-06-11T09:22:31.041211Z","shell.execute_reply.started":"2026-06-11T09:22:31.029597Z","shell.execute_reply":"2026-06-11T09:22:31.040559Z"},"jupyter":{"outputs_hidden":false}}
id2label.keys()

# %% [markdown] {"id":"EraNFBC8VnPu","jupyter":{"outputs_hidden":false}}
# Now let's encode our texts and labels!

# %% [code] {"id":"uDuGq_n4pgZX","execution":{"iopub.status.busy":"2026-06-11T09:22:31.042160Z","iopub.execute_input":"2026-06-11T09:22:31.042568Z","iopub.status.idle":"2026-06-11T09:22:34.957185Z","shell.execute_reply.started":"2026-06-11T09:22:31.042515Z","shell.execute_reply":"2026-06-11T09:22:34.956367Z"},"jupyter":{"outputs_hidden":false}}
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
test_encodings  = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

train_labels_encoded = [label2id[y] for y in train_labels]
test_labels_encoded  = [label2id[y] for y in test_labels]

#Bhoopendra Kumar,11-06-26,Code added to prepared dataset
# Save processed dataset locally

processed_train = pd.DataFrame({
    "text": train_texts,
    "label": train_labels,
    "label_id": train_labels_encoded
})

processed_test = pd.DataFrame({
    "text": test_texts,
    "label": test_labels,
    "label_id": test_labels_encoded
})

processed_train.to_csv("processed_train.csv", index=False)
processed_test.to_csv("processed_test.csv", index=False)

print("Processed datasets saved locally")

# %% [markdown] {"id":"7X1sYEGsWDDh","jupyter":{"outputs_hidden":false}}
# **Examine a Goodreads review in the training set after encoding**

# %% [code] {"id":"4A89SN_ppiUP","outputId":"1e113360-1547-4569-f1cb-b430a003b667","execution":{"iopub.status.busy":"2026-06-11T09:22:34.958180Z","iopub.execute_input":"2026-06-11T09:22:34.958546Z","iopub.status.idle":"2026-06-11T09:22:34.964090Z","shell.execute_reply.started":"2026-06-11T09:22:34.958519Z","shell.execute_reply":"2026-06-11T09:22:34.963251Z"},"jupyter":{"outputs_hidden":false}}
' '.join(train_encodings[0].tokens[0:100])

# %% [markdown] {"id":"3hvOn9RGWUMm","jupyter":{"outputs_hidden":false}}
# **Examine a Goodreads review in the test set after encoding**

# %% [code] {"id":"OafZQFKSwG9E","outputId":"4305cad0-0861-4c46-db45-cb7042e7aec2","execution":{"iopub.status.busy":"2026-06-11T09:22:34.965173Z","iopub.execute_input":"2026-06-11T09:22:34.965502Z","iopub.status.idle":"2026-06-11T09:22:34.978294Z","shell.execute_reply.started":"2026-06-11T09:22:34.965467Z","shell.execute_reply":"2026-06-11T09:22:34.977659Z"},"jupyter":{"outputs_hidden":false}}
' '.join(test_encodings[0].tokens[0:100])

# %% [markdown] {"id":"1jmt15FvW8FR","jupyter":{"outputs_hidden":false}}
# **Examine the training labels after encoding**

# %% [code] {"id":"ciemdVYwwMNz","outputId":"8af20c5c-1b54-4c39-d1e4-efcfa0b8180e","execution":{"iopub.status.busy":"2026-06-11T09:22:34.979375Z","iopub.execute_input":"2026-06-11T09:22:34.979706Z","iopub.status.idle":"2026-06-11T09:22:34.991849Z","shell.execute_reply.started":"2026-06-11T09:22:34.979682Z","shell.execute_reply":"2026-06-11T09:22:34.991037Z"},"jupyter":{"outputs_hidden":false}}
set(train_labels_encoded)

# %% [markdown] {"id":"UK1Ngb0wXBz9","jupyter":{"outputs_hidden":false}}
# **Examine the test labels after encoding**

# %% [code] {"id":"TowwulYQwOff","outputId":"46d4943b-684e-40a4-b00f-c3dc8e09e2b9","execution":{"iopub.status.busy":"2026-06-11T09:22:34.992812Z","iopub.execute_input":"2026-06-11T09:22:34.993082Z","iopub.status.idle":"2026-06-11T09:22:35.006396Z","shell.execute_reply.started":"2026-06-11T09:22:34.993060Z","shell.execute_reply":"2026-06-11T09:22:35.005636Z"},"jupyter":{"outputs_hidden":false}}
set(test_labels_encoded)

# %% [markdown] {"id":"ChcEv01TXI7v","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Make a custom Torch dataset**

# %% [markdown] {"id":"PxWcyj0LXVtY","jupyter":{"outputs_hidden":false}}
# Here we combine the encoded labels and texts into dataset objects. We use the custom Torch `MyDataSet` class to make a `train_dataset` object from  the `train_encodings` and `train_labels_encoded`. We also make a `test_dataset` object from `test_encodings`, and `test_labels_encoded`.

# %% [code] {"id":"S4VCU-nepnqF","execution":{"iopub.status.busy":"2026-06-11T09:22:35.011093Z","iopub.execute_input":"2026-06-11T09:22:35.011446Z","iopub.status.idle":"2026-06-11T09:22:35.019722Z","shell.execute_reply.started":"2026-06-11T09:22:35.011423Z","shell.execute_reply":"2026-06-11T09:22:35.018901Z"},"jupyter":{"outputs_hidden":false}}
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# %% [code] {"id":"xyCFH3XEi4Ng","execution":{"iopub.status.busy":"2026-06-11T09:22:35.020700Z","iopub.execute_input":"2026-06-11T09:22:35.021238Z","iopub.status.idle":"2026-06-11T09:22:35.037898Z","shell.execute_reply.started":"2026-06-11T09:22:35.021212Z","shell.execute_reply":"2026-06-11T09:22:35.037094Z"},"jupyter":{"outputs_hidden":false}}
train_dataset = MyDataset(train_encodings, train_labels_encoded)
test_dataset = MyDataset(test_encodings, test_labels_encoded)

# %% [markdown] {"id":"lSoXOmDYYyAK","jupyter":{"outputs_hidden":false}}
# **Examine a Goodreads review in the Torch `training_dataset` after encoding**

# %% [code] {"id":"DbJerEgC1Qpc","outputId":"5234df23-8751-44cc-d4d0-8f018f91c624","execution":{"iopub.status.busy":"2026-06-11T09:22:35.038799Z","iopub.execute_input":"2026-06-11T09:22:35.039161Z","iopub.status.idle":"2026-06-11T09:22:35.052179Z","shell.execute_reply.started":"2026-06-11T09:22:35.039106Z","shell.execute_reply":"2026-06-11T09:22:35.051027Z"},"jupyter":{"outputs_hidden":false}}
' '.join(train_dataset.encodings[0].tokens[0:100])

# %% [markdown] {"id":"hq1M2Et4Y3LB","jupyter":{"outputs_hidden":false}}
# **Examine a Goodreads review in the Torch `test_dataset` after encoding**

# %% [code] {"id":"z65jnjVJ1aVB","outputId":"5561fe65-84f8-4b77-e846-13032bc14028","execution":{"iopub.status.busy":"2026-06-11T09:22:35.053191Z","iopub.execute_input":"2026-06-11T09:22:35.053547Z","iopub.status.idle":"2026-06-11T09:22:35.064574Z","shell.execute_reply.started":"2026-06-11T09:22:35.053523Z","shell.execute_reply":"2026-06-11T09:22:35.063884Z"},"jupyter":{"outputs_hidden":false}}
' '.join(test_dataset.encodings[1].tokens[0:100])

# %% [markdown] {"id":"OkVgFcbCqKSu","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Load pre-trained BERT model**

# %% [markdown] {"id":"X2pSuFUVaDhP","jupyter":{"outputs_hidden":false}}
# Here we load a pre-trained DistilBERT model and send it to CUDA.
# 
# **Note:** If you decide to repeat fine-tuning after already running the following cells, make sure that you re-run this cell to re-load the original pre-trained model before fine-tuning again.

# %% [code] {"id":"a7k75REXp7UJ","outputId":"456f875b-65c2-4cad-8b5a-df2ed9761377","execution":{"iopub.status.busy":"2026-06-11T09:22:35.065385Z","iopub.execute_input":"2026-06-11T09:22:35.065678Z","iopub.status.idle":"2026-06-11T09:22:35.362025Z","shell.execute_reply.started":"2026-06-11T09:22:35.065623Z","shell.execute_reply":"2026-06-11T09:22:35.361244Z"},"jupyter":{"outputs_hidden":false}}
#Bhoopendra Kumar,11-06-26,code commented
# The model_name needs to match the name used for the tokenizer above.
#model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device_name)

#Bhoopendra Kumar,11-06-26,New code added to load model with id2label mapping
# Load model using label mappings

model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
).to(device_name)

print("Model loaded successfully")
print("Number of labels:", len(id2label))

# %% [markdown] {"id":"VRQZuqcAqQNI","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Set the BERT fine-tuning parameters**
# 
# These are the arguments we'll set in the HuggingFace TrainingArguments objects, which we'll then pass to the HuggingFace Trainer object. There are many more possible arguments, but here we highlight the basics and some common gotchas.
# 
# When training your own model, you should search over these parameters to find the best settings for your particular dataset. You should use a held-out set of validation data for this step.

# %% [markdown] {"id":"yYOaH9AhbCD_","jupyter":{"outputs_hidden":false}}
# | Parameter | Explanation |
# |-----------| ------------|
# | num_train_epochs | total number of training epochs (how many times to pass through the entire dataset; too much can cause overfitting) |
# | per_device_train_batch_size | batch size per device during training |
# | per_device_eval_batch_size |  batch size for evaluation |
# |  warmup_steps |  number of warmup steps for learning rate scheduler (set lower because of small dataset size) |
# | weight_decay | strength of weight decay (reduces size of weights, like regularization) |
# | output_dir | output directory for the fine-tuned model and configuration files |
# | logging_dir | directory for storing logs |
# | logging_steps | how often to print logging output (so that we can stop training early if the loss isn't going down) |
# | evaluation_strategy | evaluate while training so that we can see the accuracy going up |

# %% [markdown] {"id":"8Pb3xtidn-HJ","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Fine-tune the BERT model**

# %% [code] {"id":"3idCswBVg6v_","outputId":"bbba556c-da16-4da5-904d-bcfc405f93dd","execution":{"iopub.status.busy":"2026-06-11T09:22:35.363028Z","iopub.execute_input":"2026-06-11T09:22:35.363752Z","iopub.status.idle":"2026-06-11T09:40:09.873320Z","shell.execute_reply.started":"2026-06-11T09:22:35.363725Z","shell.execute_reply":"2026-06-11T09:40:09.872389Z"},"jupyter":{"outputs_hidden":false}}
# Before training, initialize wandb
import wandb
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
 
# 1. Initialise W&B
wandb.init(
    project="mlops-group27-assignment",
    name="run-v2",
	config={
    	"model": model_name,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "max_length": max_length,
        "dataset": "UCSD Goodreads",
        "platform": "Kaggle",
	}
)
 
# 2. Define evaluation metrics
def compute_metrics(pred):
	labels = pred.label_ids
	preds  = pred.predictions.argmax(-1)
	return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
	}
 
# 3. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
	num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
	warmup_steps=100,
	weight_decay=0.01,
	logging_steps=50,
    eval_strategy="epoch",
	save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="run-v2",
)
 
# 4. Train
trainer = Trainer(
	model=model,
	args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
 
# 5. End W&B run
wandb.finish()

# %% [markdown] {"id":"z_SN_oGLV8Vw","jupyter":{"outputs_hidden":false}}
# First, we define a custom evaluation function that returns the accuracy. You could modify this function to return precision, recall, F1, and/or other metrics.

# %% [code] {"id":"pNIt7fcnqUCp","execution":{"iopub.status.busy":"2026-06-11T09:40:09.874585Z","iopub.execute_input":"2026-06-11T09:40:09.874882Z","iopub.status.idle":"2026-06-11T09:40:09.880011Z","shell.execute_reply.started":"2026-06-11T09:40:09.874855Z","shell.execute_reply":"2026-06-11T09:40:09.879201Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# %% [markdown] {"id":"g9xl5QLmWsAw","jupyter":{"outputs_hidden":false}}
# Then we create a HuggingFace `Trainer` object using the `TrainingArguments` object that we created above. We also send our `compute_metrics` function to the `Trainer` object, along with our test and train datasets.
# 
# **Note:** This is what we've been aiming for this whole time! All the work of tokenizing, creating datasets, and setting the training arguments was for this cell.

# %% [code] {"id":"fgc8FS50qV0_","execution":{"iopub.status.busy":"2026-06-11T09:40:09.880979Z","iopub.execute_input":"2026-06-11T09:40:09.881797Z","iopub.status.idle":"2026-06-11T09:40:09.900743Z","shell.execute_reply.started":"2026-06-11T09:40:09.881772Z","shell.execute_reply":"2026-06-11T09:40:09.899939Z"},"jupyter":{"outputs_hidden":false}}
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset (usually a validation set; here we just send our test set)
    compute_metrics=compute_metrics      # our custom evaluation function
)

# %% [markdown] {"id":"Mo5QVLYjXGCN","jupyter":{"outputs_hidden":false}}
# Time to finally fine-tune!
# 
# Be patient; if you've set everything in Colab to use GPUs, then it should only take a minute or two to run, but if you're running on CPU, it can take hours.
# 
# After every 10 steps (as we specified in the TrainingArguments object), the trainer will output the current state of the model, including the training loss, validation ("test") loss, and accuracy (from our `compute_metrics` function).
# 
# You should see the loss going down and the accuracy going up. If instead they are staying the same or oscillating, you probably need to change the fine-tuning parameters.

# %% [code] {"id":"Od9TxpEj4tBZ","execution":{"iopub.status.busy":"2026-06-11T09:40:09.901694Z","iopub.execute_input":"2026-06-11T09:40:09.902032Z","iopub.status.idle":"2026-06-11T09:40:09.905958Z","shell.execute_reply.started":"2026-06-11T09:40:09.901996Z","shell.execute_reply":"2026-06-11T09:40:09.905345Z"},"jupyter":{"outputs_hidden":false}}
# Turn off weights and biases logging, which requires an API key

import os
os.environ["WANDB_DISABLED"] = "false"

# %% [code] {"id":"W64JwriVqcmk","outputId":"c8ac8d66-0a02-4841-a175-e085c2d4030b","execution":{"iopub.status.busy":"2026-06-11T09:40:09.906760Z","iopub.execute_input":"2026-06-11T09:40:09.907275Z","iopub.status.idle":"2026-06-11T09:57:47.523721Z","shell.execute_reply.started":"2026-06-11T09:40:09.907251Z","shell.execute_reply":"2026-06-11T09:57:47.522862Z"},"jupyter":{"outputs_hidden":false}}
trainer.train()

# %% [markdown] {"id":"kXeIZ_LFqeps","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Save fine-tuned model**
# 
# The following cell will save the model and its configuration files to a directory in Colab. To preserve this model for future use, you should download the model to your computer.

# %% [code] {"id":"kxkDWDfvqeAo","execution":{"iopub.status.busy":"2026-06-11T09:57:47.524757Z","iopub.execute_input":"2026-06-11T09:57:47.525105Z","iopub.status.idle":"2026-06-11T09:57:48.178494Z","shell.execute_reply.started":"2026-06-11T09:57:47.525068Z","shell.execute_reply":"2026-06-11T09:57:48.177877Z"},"jupyter":{"outputs_hidden":false}}
trainer.save_model(cached_model_directory_name)

#Bhoopendra Kumar, 11-06-26, Code added to upload prepared data into hugging face
import shutil

shutil.copy(
    "id2label.json",
    f"{cached_model_directory_name}/id2label.json"
)

# %% [markdown] {"id":"epiLftYkZrzc","jupyter":{"outputs_hidden":false}}
# (Optional) If you've already fine-tuned and saved the model, you can reload it using the following line. You don't have to run fine-tuning every time you want to evaluate.

# %% [code] {"id":"9A54QySLrO5I","execution":{"iopub.status.busy":"2026-06-11T09:57:48.179598Z","iopub.execute_input":"2026-06-11T09:57:48.180146Z","iopub.status.idle":"2026-06-11T09:57:48.184642Z","shell.execute_reply.started":"2026-06-11T09:57:48.180101Z","shell.execute_reply":"2026-06-11T09:57:48.183903Z"},"jupyter":{"outputs_hidden":false}}
# trainer = DistilBertForSequenceClassification.from_pretrained(cached_model_directory_name)

# %% [markdown] {"id":"JpzV4hFsLmZ6","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Evaluate fine-tuned model**

# %% [markdown] {"id":"-IvfhrBtYYcz","jupyter":{"outputs_hidden":false}}
# The following function of the `Trainer` object will run the built-in evaluation, including our `compute_metrics` function.

# %% [code] {"id":"dshtTH0WLtM1","execution":{"iopub.status.busy":"2026-06-11T09:57:48.185631Z","iopub.execute_input":"2026-06-11T09:57:48.185942Z","iopub.status.idle":"2026-06-11T09:58:15.299975Z","shell.execute_reply.started":"2026-06-11T09:57:48.185910Z","shell.execute_reply":"2026-06-11T09:58:15.299158Z"},"jupyter":{"outputs_hidden":false}}
trainer.evaluate()

# %% [markdown] {"id":"ILJGLcCjYhPt","jupyter":{"outputs_hidden":false}}
# But we might want to do more fine-grained analysis of the model, so we extract the predicted labels.

# %% [code] {"id":"v_E8oVjeLuv2","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:15.300902Z","iopub.execute_input":"2026-06-11T09:58:15.301113Z","iopub.status.idle":"2026-06-11T09:58:42.262220Z","shell.execute_reply.started":"2026-06-11T09:58:15.301091Z","shell.execute_reply":"2026-06-11T09:58:42.261606Z"}}
predicted_results = trainer.predict(test_dataset)

# %% [code] {"id":"WUYGfzczOuJE","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.263186Z","iopub.execute_input":"2026-06-11T09:58:42.263503Z","iopub.status.idle":"2026-06-11T09:58:42.269883Z","shell.execute_reply.started":"2026-06-11T09:58:42.263479Z","shell.execute_reply":"2026-06-11T09:58:42.269022Z"}}
predicted_results.predictions.shape

# %% [code] {"id":"hqUTa5irLyN8","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.270948Z","iopub.execute_input":"2026-06-11T09:58:42.271379Z","iopub.status.idle":"2026-06-11T09:58:42.285658Z","shell.execute_reply.started":"2026-06-11T09:58:42.271352Z","shell.execute_reply":"2026-06-11T09:58:42.284938Z"}}
predicted_labels = predicted_results.predictions.argmax(-1) # Get the highest probability prediction
predicted_labels = predicted_labels.flatten().tolist()      # Flatten the predictions into a 1D list
predicted_labels = [id2label[l] for l in predicted_labels]  # Convert from integers back to strings for readability

# %% [code] {"id":"Y2jtqnJbPMpu","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.286484Z","iopub.execute_input":"2026-06-11T09:58:42.286742Z","iopub.status.idle":"2026-06-11T09:58:42.299335Z","shell.execute_reply.started":"2026-06-11T09:58:42.286719Z","shell.execute_reply":"2026-06-11T09:58:42.298560Z"}}
len(predicted_labels)

# %% [code] {"id":"qVcMU45fLzli","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.300365Z","iopub.execute_input":"2026-06-11T09:58:42.300703Z","iopub.status.idle":"2026-06-11T09:58:42.345975Z","shell.execute_reply.started":"2026-06-11T09:58:42.300650Z","shell.execute_reply":"2026-06-11T09:58:42.345382Z"}}
print(classification_report(test_labels,
                            predicted_labels))

# %% [markdown] {"id":"ddk-iKTQF-K3","jupyter":{"outputs_hidden":false}}
# <br><br>
# 
# ## **Pull out correct and incorrect classifications for examination**
# 
# Let's use our predicted labels for some analysis!
# 
# Now that we've fine-tuned and pulled out our predicted labels, the BERT part of this tutorial is done. You can now use the predicted labels in the same way you would use any set of predicted labels from any classification model. We'll show some examples here.
# 
# First, let's print out some example predictions that were correct.

# %% [code] {"id":"l7KrvGuZkDAV","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.346829Z","iopub.execute_input":"2026-06-11T09:58:42.347107Z","iopub.status.idle":"2026-06-11T09:58:42.373618Z","shell.execute_reply.started":"2026-06-11T09:58:42.347071Z","shell.execute_reply":"2026-06-11T09:58:42.373067Z"}}
for _true_label, _predicted_label, _text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
  if _true_label == _predicted_label:
    print('LABEL:', _true_label)
    print('REVIEW TEXT:', _text[:100], '...')
    print()

# %% [markdown] {"id":"pW30Z6ynkDPI","jupyter":{"outputs_hidden":false}}
# Now let's print out some misclassifications.

# %% [code] {"id":"Xmx1RSKDkIpG","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.374514Z","iopub.execute_input":"2026-06-11T09:58:42.374934Z","iopub.status.idle":"2026-06-11T09:58:42.389717Z","shell.execute_reply.started":"2026-06-11T09:58:42.374891Z","shell.execute_reply":"2026-06-11T09:58:42.388896Z"}}
for _true_label, _predicted_label, _text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
  if _true_label != _predicted_label:
    print('TRUE LABEL:', _true_label)
    print('PREDICTED LABEL:', _predicted_label)
    print('REVIEW TEXT:', _text[:100], '...')
    print()

# %% [markdown] {"id":"3MZqyFrckJBB","jupyter":{"outputs_hidden":false}}
# Finally, let's create some heatmaps to examine misclassification patterns. We could use these patterns to think about similarities and differences between genres, according to book reviewers.

# %% [code] {"id":"v8yJ3-Z7hLXf","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.391241Z","iopub.execute_input":"2026-06-11T09:58:42.391899Z","iopub.status.idle":"2026-06-11T09:58:42.404550Z","shell.execute_reply.started":"2026-06-11T09:58:42.391860Z","shell.execute_reply":"2026-06-11T09:58:42.403644Z"}}
genre_classifications_dict = defaultdict(int)
for _true_label, _predicted_label in zip(test_labels, predicted_labels):
  genre_classifications_dict[(_true_label, _predicted_label)] += 1

dicts_to_plot = []
for (_true_genre, _predicted_genre), _count in genre_classifications_dict.items():
  dicts_to_plot.append({'True Genre': _true_genre,
                        'Predicted Genre': _predicted_genre,
                        'Number of Classifications': _count})

df_to_plot = pd.DataFrame(dicts_to_plot)
df_wide = df_to_plot.pivot_table(index='True Genre',
                                 columns='Predicted Genre',
                                 values='Number of Classifications')

# %% [code] {"id":"wSAgS6tvivvz","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.405492Z","iopub.execute_input":"2026-06-11T09:58:42.405914Z","iopub.status.idle":"2026-06-11T09:58:42.625133Z","shell.execute_reply.started":"2026-06-11T09:58:42.405884Z","shell.execute_reply":"2026-06-11T09:58:42.624558Z"}}
plt.figure(figsize=(9,7))
sns.set(style='ticks', font_scale=1.2)
sns.heatmap(df_wide, linewidths=1, cmap='Purples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown] {"id":"eiukJdYRqyjm","jupyter":{"outputs_hidden":false}}
# Looks good! We can see that overall, our model is assigning the correct labels for each genre.
# 
# Now, let's remove the diagonal from the plot to highlight the misclassifications.

# %% [code] {"id":"O7GJQfYxi3ET","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.626051Z","iopub.execute_input":"2026-06-11T09:58:42.626490Z","iopub.status.idle":"2026-06-11T09:58:42.638952Z","shell.execute_reply.started":"2026-06-11T09:58:42.626461Z","shell.execute_reply":"2026-06-11T09:58:42.638250Z"}}
genre_classifications_dict = defaultdict(int)
for _true_label, _predicted_label in zip(test_labels, predicted_labels):
  if _true_label != _predicted_label: # Remove the diagonal to highlight misclassifications
    genre_classifications_dict[(_true_label, _predicted_label)] += 1

dicts_to_plot = []
for (_true_genre, _predicted_genre), _count in genre_classifications_dict.items():
  dicts_to_plot.append({'True Genre': _true_genre,
                        'Predicted Genre': _predicted_genre,
                        'Number of Classifications': _count})

df_to_plot = pd.DataFrame(dicts_to_plot)
df_wide = df_to_plot.pivot_table(index='True Genre',
                                 columns='Predicted Genre',
                                 values='Number of Classifications')

# %% [code] {"id":"3oM09JGrjX_y","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.639780Z","iopub.execute_input":"2026-06-11T09:58:42.640060Z","iopub.status.idle":"2026-06-11T09:58:42.843238Z","shell.execute_reply.started":"2026-06-11T09:58:42.640037Z","shell.execute_reply":"2026-06-11T09:58:42.842604Z"}}
plt.figure(figsize=(9,7))
sns.set(style='ticks', font_scale=1.2)
sns.heatmap(df_wide, linewidths=1, cmap='Purples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown] {"id":"w99vVZCVl8g0","jupyter":{"outputs_hidden":false}}
# There's much more you can do with your own dataset and labels! Classification can be used to apply a small set of labels across a big dataset; to explore misclassifications to better understand users; and much more! We hope you'll use this tutorial in all kinds of creative ways.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **Evaluation Script for Reference!**

# %% [code] {"id":"ZEnxI21djZXT","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:58:42.844191Z","iopub.execute_input":"2026-06-11T09:58:42.844489Z","iopub.status.idle":"2026-06-11T09:59:39.408119Z","shell.execute_reply.started":"2026-06-11T09:58:42.844453Z","shell.execute_reply":"2026-06-11T09:59:39.407234Z"}}
import json, wandb
from sklearn.metrics import classification_report
 
# Run evaluation
eval_results = trainer.evaluate()
print(eval_results)
 
# Log final metrics to W&B
wandb.log({
    "final/loss":     eval_results["eval_loss"],
    "final/accuracy": eval_results["eval_accuracy"],
    "final/f1":       eval_results["eval_f1"],
})
 
# Save full classification report
preds  = trainer.predict(test_dataset).predictions.argmax(-1)
labels = [item["labels"].item() for item in test_dataset]
report = classification_report(
	labels, preds, target_names=list(id2label.values()), output_dict=True
)
 
with open("eval_report.json", "w") as f:
	json.dump(report, f, indent=2)
 
# Upload to W&B as a versioned Artifact
artifact = wandb.Artifact("eval-report", type="evaluation")
artifact.add_file("eval_report.json")
wandb.log_artifact(artifact)
wandb.finish()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-11T09:59:39.409289Z","iopub.execute_input":"2026-06-11T09:59:39.410186Z","iopub.status.idle":"2026-06-11T09:59:52.211241Z","shell.execute_reply.started":"2026-06-11T09:59:39.410119Z","shell.execute_reply":"2026-06-11T09:59:52.210422Z"}}
from huggingface_hub import login
 
# HF_TOKEN was already loaded from Kaggle Secrets (see Task 4)
login(token=HF_TOKEN)
 
# Push model and tokenizer
model.push_to_hub("Bhoop-g25ait2025/distilbert-goodreads-genres_g27")
tokenizer.push_to_hub("Bhoop-g25ait2025/distilbert-goodreads-genres_g27")

#Bhoopendra Kumar,11-06-26,code added to upload id2label.json into hugging face
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="id2label.json",
    path_in_repo="id2label.json",
    repo_id="Bhoop-g25ait2025/distilbert-goodreads-genres_g27",
    repo_type="model"
)

print("id2label.json uploaded to Hugging Face")
 
# Reinitialize W&B to record the link (since wandb.finish() was called earlier)
wandb.init(project="mlops-group27-assignment", name="run-v2", resume="allow")
wandb.run.summary["mlops-group27-assignment"] = \
    "https://huggingface.co/Bhoop-g25ait2025/distilbert-goodreads-genres_g27"
wandb.finish()
