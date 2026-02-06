from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime

class ImageData(BaseModel):
    fundus: HttpUrl
    oct: Optional[HttpUrl] = None
    
class TabularData(BaseModel):
    age: int
    sex: int
    dm_time: float
    alcohol_consumption: bool
    smoking: bool
    
class MetaData(BaseModel):
    submission_time: datetime
    submitted_by: str

class DRPredictionRequest(BaseModel):
    patient_id: str
    image_data: ImageData
    tabular_data: Optional[TabularData] = None
    metadata: MetaData