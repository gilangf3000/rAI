import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_excel("./datasets/rAI-beta.xlsx")

image_texts = df["IMAGE"].dropna().tolist()
text_texts = df["TEXT"].dropna().tolist()

texts = image_texts + text_texts
labels = ["IMAGE"] * len(image_texts) + ["TEXT"] * len(text_texts)

emb = model.encode(texts, normalize_embeddings=True).astype("float16")

np.savez_compressed(
    "models/rAI-beta.npz",
    embeddings=emb,
    labels=np.array(labels),
    texts=np.array(texts)
)

print("saved rAI-beta.npz")