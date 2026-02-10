import sys
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add backend directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.app.core.config import settings

def train():
    print("Starting training process...")
    
    # Load dataset
    # Local datasets folder in training
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "rAI-beta.xlsx")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please run scripts/create_dataset.py first.")
        return

    df = pd.read_excel(DATASET_PATH)
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")

    
    # Prepare data
    X_text = df['text'].tolist()
    y = df['label'].tolist()
    
    # Initialize embedding model
    print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    encoder = SentenceTransformer(settings.EMBEDDING_MODEL, device='cpu')

    # Quantize for training script memory as well
    try:
        import torch
        from torch.quantization import quantize_dynamic
        print("Optimizing model for low RAM...")
        encoder[0].auto_model = quantize_dynamic(
            encoder[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as e:
        print(f"Quantization warning: {e}")
    
    # Generate embeddings
    print("Generating embeddings...")
    X = encoder.encode(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    print("Training SGDClassifier...")
    classifier = SGDClassifier(loss='log_loss', random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model to BACKEND models directory
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "models", "image_classifier.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
