import os
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from app.core.config import settings
from app.core.schemas import PredictionLabel

class ImageDetectionModel:
    def __init__(self):
        self.labels = [label.value for label in PredictionLabel]
        self.model_path = settings.MODEL_PATH
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load the pipeline (Vectorizer + Classifier) from disk."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Starting with fresh pipeline.")
                self.init_fresh_pipeline()
        else:
            print("No existing model found. Starting with fresh pipeline.")
            self.init_fresh_pipeline()

    def init_fresh_pipeline(self):
        """Initialize a fresh Pipeline with TF-IDF Vectorizer and SGD Classifier."""
        # TF-IDF Configuration for multilingual robustness
        # analyzer='char_wb' helps with typos and morphological variations
        # ngram_range=(2, 5) captures short phrases
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=1, max_features=50000)),
            ('clf', SGDClassifier(loss='log_loss', random_state=42))
        ])

    def save_model(self):
        """Save the pipeline to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, text: str):
        """
        Predict if the text is an image generation request.
        Returns: (Label, Confidence)
        """
        if not self.pipeline:
             return "TEXT", 0.0

        try:
            # Predict probabilities
            probs = self.pipeline.predict_proba([text])[0]
            
            # Get class labels from the classifier step
            classes = self.pipeline.named_steps['clf'].classes_
            
            top_class_index = np.argmax(probs)
            label = classes[top_class_index]
            confidence = probs[top_class_index]
            
            return label, float(confidence)
            
        except NotFittedError:
            print("Model not fitted yet or needs retraining.")
            return "TEXT", 0.0
        except Exception as e:
            print(f"Prediction error: {e}")
            return "TEXT", 0.0

    def learn(self, text: str, label: str):
        """
        Online learning: update the classifier with a new example.
        Note: TfidfVectorizer vocabulary is fixed after initial fit.
              New words not in vocab will be ignored, but weights for existing n-grams will update.
        """
        if not self.pipeline:
            self.init_fresh_pipeline()
            
        # Get the vectorizer and classifier steps
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['clf']
        
        # Transform input text
        try:
             # Transform using existing vocabulary
            X_vector = vectorizer.transform([text])
            
            # Partial fit the classifier
            # If classes are not set (fresh model), provide them
            if not hasattr(classifier, "classes_"):
                 classifier.partial_fit(X_vector, [label], classes=self.labels)
            else:
                classifier.partial_fit(X_vector, [label])
                
            self.save_model()
            return True
        except NotFittedError:
             # If vectorizer is not fitted, we can't transform.
             # This happens if 'init_fresh_pipeline' was called but never 'fit'.
             # In a real scenario, we should have a base model pre-trained.
             print("Vectorized not fitted. Cannot learn online without initial pre-training.")
             return False

# Global instance
model = ImageDetectionModel()
