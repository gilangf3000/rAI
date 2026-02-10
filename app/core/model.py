import os
import pickle
import numpy as np
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import NotFittedError

from app.core.config import settings

class ImageDetectionModel:
    def __init__(self):
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.classifier = SGDClassifier(loss='log_loss', random_state=42)
        self.labels = ["TEXT", "IMAGE"]
        self.model_path = settings.MODEL_PATH
        
        self.load_model()

    def load_model(self):
        """Load the classifier from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Starting with fresh model.")
        else:
            print("No existing model found. Starting with fresh model.")

    def save_model(self):
        """Save the classifier to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {self.model_path}")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if the text is an image generation request.
        Returns: (Label, Confidence)
        """
        embedding = self.encoder.encode([text])
        
        try:
            probs = self.classifier.predict_proba(embedding)[0]
            # Assumes classifier.classes_ are sorted, usually ['IMAGE', 'TEXT'] or similar depending on training
            # We need to map probability to the correct label.
            # Best way is to use classifier.classes_
            
            top_class_index = np.argmax(probs)
            label = self.classifier.classes_[top_class_index]
            confidence = probs[top_class_index]
            
            return label, float(confidence)
            
        except NotFittedError:
            print("Model not fitted yet or needs retraining.")
            # Fallback or default behavior if model isn't trained
            return "TEXT", 0.0

    def learn(self, text: str, label: str):
        """
        Online learning: update the model with a new example.
        """
        embedding = self.encoder.encode([text])
        
        # Partial fit expects arrays
        # If this is the very first fit, we need to provide classes
        if not hasattr(self.classifier, "classes_"):
             self.classifier.partial_fit(embedding, [label], classes=self.labels)
        else:
            self.classifier.partial_fit(embedding, [label])
            
        self.save_model()
        return True

# Global instance
model = ImageDetectionModel()
