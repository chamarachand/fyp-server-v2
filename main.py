from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.fundus_preprocessing import preprocess_local_image, preprocess_upload
from schemas.dr_schema import DRPredictionRequest
from routes import prediction_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to "*" temporarily to make Postman testing easier
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "DR Diagnosis API is live!", "docs": "/docs"}

app.include_router(prediction_routes.router)
