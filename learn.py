import numpy as np
import json

N_TRIALS = 200
npz_file = "models/rai.npz"

data = np.load(npz_file, allow_pickle=True)
emb = data["embeddings"].astype("float32")  # float32 lebih stabil
labels = data["labels"]
texts = data["texts"]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

best_score = -1
best_centroid = None

for trial in range(N_TRIALS):
    idx = np.random.permutation(len(labels))
    emb_shuffled = emb[idx]
    labels_shuffled = labels[idx]

    idx_img = [i for i, l in enumerate(labels_shuffled) if l == "IMAGE"]
    idx_txt = [i for i, l in enumerate(labels_shuffled) if l == "TEXT"]

    centroid_image = emb_shuffled[idx_img].mean(axis=0)
    centroid_text  = emb_shuffled[idx_txt].mean(axis=0)

    correct = 0
    for e, l in zip(emb_shuffled, labels_shuffled):
        pred = "IMAGE" if cosine(e, centroid_image) > cosine(e, centroid_text) else "TEXT"
        if pred == l:
            correct += 1

    acc = correct / len(labels)
    if acc > best_score:
        best_score = acc
        best_centroid = {
            "IMAGE": centroid_image.tolist(),
            "TEXT": centroid_text.tolist()
        }

print(f"Best accuracy after {N_TRIALS} trials: {best_score:.4f}")

out = {
    "model": "all-MiniLM-L6-v2",
    "labels": ["IMAGE", "TEXT"],
    "centroids": best_centroid
}

with open("models/rAI-beta.json", "w") as f:
    json.dump(out, f)

print("Saved best centroids to models/rAI-beta.json")
