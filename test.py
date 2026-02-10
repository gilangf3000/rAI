import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load centroid JSON hasil learn.py
with open("models/rAI-beta.json") as f:
    db = json.load(f)

MODEL_NAME = db["model"]
model = SentenceTransformer(MODEL_NAME)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect(msg):
    # encode pesan user
    emb = model.encode([msg], normalize_embeddings=True)[0]
    score_image = cosine(emb, np.array(db["centroids"]["IMAGE"]))
    score_text  = cosine(emb, np.array(db["centroids"]["TEXT"]))
    return "IMAGE" if score_image > score_text else "TEXT"

print("AI Router CLI (ketik 'exit' untuk keluar)\n")

while True:
    msg = input(">>> ")
    if msg.lower() == "exit":
        break

    intent = detect(msg)
    if intent == "IMAGE":
        print(f"[IMAGE ROUTE] → {msg}\n")
    else:
        print(f"[TEXT ROUTE] → {msg}\n")
