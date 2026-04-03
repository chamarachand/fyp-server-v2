from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl
from datetime import datetime

class PredictionTypes(BaseModel):
    combined_predictions: Dict

class SavePredictionRequest(BaseModel):
    doctor_id: str
    patient_name: str
    patient_id: Optional[str] = None
    prediction: PredictionTypes
    fundus_image_url: Optional[str] = None
    oct_image_url: Optional[str] = None
    health_data: Dict = None
    
