import os

class Settings:
    PROJECT_NAME: str = "AI Router"
    VERSION: str = "0.1.0"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MODEL_PATH: str = "models/image_classifier.pkl"
    CONFIDENCE_THRESHOLD: float = 0.6
    
    # Training Configuration
    DATASET_PATH: str = "datasets/rAI-beta.xlsx"

settings = Settings()
