import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

df = pd.read_excel("./datasets/rAI-beta.xlsx")

image_texts = df["IMAGE"].dropna().tolist()
text_texts = df["TEXT"].dropna().tolist()

def encode(texts):
    return model.encode(
        texts,
        normalize_embeddings=True
    ).astype("float32")


img_emb = encode(image_texts)
txt_emb = encode(text_texts)

centroid_image = img_emb.mean(axis=0)
centroid_text = txt_emb.mean(axis=0)

out = {
    "model": MODEL_NAME,
    "labels": ["IMAGE", "TEXT"],
    "centroids": {
        "IMAGE": centroid_image.tolist(),
        "TEXT": centroid_text.tolist()
    }
}

with open("models/rAI-beta.json", "w") as f:
    json.dump(out, f)

print("saved models/rAI-beta.json")
