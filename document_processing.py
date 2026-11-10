# document_processing.py
"""Loading the WANDS product collection and text preprocessing."""
import os
import re
import urllib.request

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


DATA_DIR = "wands"
BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/"
FILES = ["product.csv", "query.csv", "label.csv"]

# Product fields and the column of product.csv each one comes from
FIELDS = ["name", "class", "category", "description", "features"]
COLUMNS = {"name": "product_name", "class": "product_class", "category": "category hierarchy",
           "description": "product_description", "features": "product_features"}


def download_wands(data_dir=DATA_DIR):
    """Download the three WANDS files once (skipped if they already exist)."""
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(BASE_URL + name, path)


def load_products(data_dir=DATA_DIR):
    """Return the product table and the list of product ids (row i of the index = doc_ids[i])."""
    products = pd.read_csv(f"{data_dir}/product.csv", sep="\t", dtype=str).fillna("")
    # features are stored as "attr:value|attr:value"; turn the separators into spaces
    products["product_features"] = products["product_features"].str.replace("|", " ").str.replace(":", " ")
    doc_ids = products["product_id"].tolist()
    return products, doc_ids


# preprocessing
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()
stem_cache = {}


def preprocess(text):
    """Lower-case, regex tokenisation, stop-word removal, Porter stemming."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOP_WORDS or len(t) < 2:
            continue
        if t not in stem_cache:
            stem_cache[t] = stemmer.stem(t)
        out.append(stem_cache[t])
    return out
