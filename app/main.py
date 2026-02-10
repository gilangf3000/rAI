from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from app.core.model import model
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class FeedbackRequest(BaseModel):
    text: str
    label: str # "IMAGE" or "TEXT"

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
    if request.label not in ["IMAGE", "TEXT"]:
        raise HTTPException(status_code=400, detail="Invalid label. Must be IMAGE or TEXT")
    
    try:
        model.learn(request.text, request.label)
        return {"status": "learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
