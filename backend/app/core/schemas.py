from enum import Enum

class PredictionLabel(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    SEARCH = "SEARCH"
