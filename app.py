from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()

# Load the best model from the pickle file
model_path = "best_model.pkl"  # Replace with the correct path to your model file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(
    feature1: float = Form(...),
    feature2: float = Form(...),
    feature3: float = Form(...),
    feature4: float = Form(...)
):
    try:
        features = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))