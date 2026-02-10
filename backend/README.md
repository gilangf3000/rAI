# AI Router - Backend

This folder contains the Python FastAPI application that serves the AI model.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize Dataset & Train Model:
   ```bash
   python scripts/create_dataset.py
   python scripts/train_initial.py
   ```

3. Run API Server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Endpoints

- **POST /predict**: `{"text": "your text here"}` -> `{"label": "IMAGE"|"TEXT", "confidence": 0.99}`
- **POST /feedback**: `{"text": "...", "label": "IMAGE"}` -> `{"status": "learned"}`
- **GET /health**: Check server status.
