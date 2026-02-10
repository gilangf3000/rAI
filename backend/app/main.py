from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from app.core.model import model
from app.core.config import settings
from app.core.schemas import PredictionLabel

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: PredictionLabel
    confidence: float

class FeedbackRequest(BaseModel):
    text: str
    label: PredictionLabel # Validates against Enum automatically

class FeedbackResponse(BaseModel):
    status: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    label, confidence = model.predict(request.text)
    return {"label": label, "confidence": confidence}

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    # Pydantic validates label is one of the Enum values
    
    try:
        model.learn(request.text, request.label.value)
        return {"status": "learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
