import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_excel("./datasets/rAI-beta.xlsx")

image_texts = df["IMAGE"].dropna().tolist()
text_texts = df["TEXT"].dropna().tolist()

def augment(texts, n=2):
    out = []
    for t in texts:
        out.append(t)
        for _ in range(n):
            out.append(t + " ")
    return out

image_texts = augment(image_texts, n=2)
text_texts = augment(text_texts, n=2)

# pastikan dataset imbang
len_img = len(image_texts)
len_txt = len(text_texts)

if len_img > len_txt:
    text_texts = np.random.choice(text_texts, size=len_img, replace=True).tolist()
elif len_txt > len_img:
    image_texts = np.random.choice(image_texts, size=len_txt, replace=True).tolist()

texts = image_texts + text_texts
labels = ["IMAGE"] * len(image_texts) + ["TEXT"] * len(text_texts)

emb = model.encode(texts, normalize_embeddings=True).astype("float16")

np.savez_compressed(
    "models/rai.npz",
    embeddings=emb,
    labels=np.array(labels),
    texts=np.array(texts)
)

print("saved models/rai.npz")
