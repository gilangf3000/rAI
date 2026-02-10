import sys
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add backend directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.app.core.config import settings

def train():
    print("Starting training process (Lightweight - No Torch)...")
    
    # Load dataset
    # Local datasets folder in training
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "rAI-beta.xlsx")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please run scripts/create_dataset.py first.")
        return

    df = pd.read_excel(DATASET_PATH)
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")
    
    # Prepare data
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Initialize Pipeline (Vectorizer + Classifier)
    # analyzer='char_wb' helps with typos and morphological variations
    # ngram_range=(2, 5) captures short phrases
    print("Initializing TF-IDF Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=1, max_features=50000)),
        ('clf', SGDClassifier(loss='log_loss', random_state=42))
    ])
    
    # Train Pipeline
    print("Training Pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model to BACKEND models directory
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "models", "image_classifier.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
