import sys
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

def train():
    print("Starting training process...")
    
    # Load dataset
    if not os.path.exists(settings.DATASET_PATH):
        print(f"Dataset not found at {settings.DATASET_PATH}. Please run scripts/create_dataset.py first.")
        return

    df = pd.read_excel(settings.DATASET_PATH)
    print(f"Loaded {len(df)} samples from {settings.DATASET_PATH}")
    
    # Prepare data
    X_text = df['text'].tolist()
    y = df['label'].tolist()
    
    # Initialize embedding model
    print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
    
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
    
    # Save model
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    with open(settings.MODEL_PATH, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {settings.MODEL_PATH}")

if __name__ == "__main__":
    train()
