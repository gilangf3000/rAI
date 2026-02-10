import numpy as np
from sentence_transformers import SentenceTransformer

data = np.load("models/rAI-beta.npz", allow_pickle=True)

embeddings = data["embeddings"]
labels = data["labels"]

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a, b):
    return np.dot(a, b)


def detect(msg):
    q = model.encode([msg], normalize_embeddings=True)[0].astype("float16")
    sims = embeddings @ q
    return labels[int(np.argmax(sims))]


print("AI Router CLI (exit untuk keluar)\n")

while True:
    msg = input(">>> ")

    if msg.lower() == "exit":
        break

    intent = detect(msg)

    if intent == "IMAGE":
        print(f"\n[IMAGE ROUTE] → {msg}\n")
    else:
        print(f"\n[TEXT ROUTE] → {msg}\n")
