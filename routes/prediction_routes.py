# routes/dr_routes.py
from fastapi import APIRouter, HTTPException
import numpy as np
import os
import requests
from io import BytesIO

from models import fundus_model, oct_model, tabular_model_dr, tabular_model_dme
from utils.fundus_preprocessing import preprocess_upload
from utils.oct_preprocessing import preprocess_oct
from utils.tabular_preprocessing import preprocess_tabular
from fusion.dr_fusion import fuse_dr_prediction
from fusion.dme_fusion import fuse_dme_prediction
from schemas.dr_schema import DRPredictionRequest

router = APIRouter(prefix="/predict")

CLASS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

OCT_LABELS = {
    0: "DME",
    1: "NORMAL",
    2: "OTHER_DISEASE"
}

FEATURE_ORDER = [
    "age",
    "sex",
    "dm_time",
    "alcohol_consumption",
    "smoking"
]

def get_prediction(model, image):
    predictions = model.predict(image)
    print('predictions', predictions)
    pred_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return predictions[0].round(4).tolist(), pred_class, confidence
    

@router.get("/home")
def home():
    return {"msg": "hello from server"}

# @router.get("/local")
# def predict_local(request: DRPredictionRequest):
#     IMAGE_PATH = "test_images/test-2.png"
    
#     if not os.path.exists(IMAGE_PATH):
#         raise HTTPException(status_code=404, detail="Image not found")
        
#     fundus_image = preprocess_local_image(IMAGE_PATH)
#     predictions, predict_class, confidence = get_prediction(fundus_model, fundus_image)
    
#     output = {
#         "image_path": IMAGE_PATH,
#         "prediction": CLASS_LABELS[predict_class],
#         "class_index": predict_class,
#         "confidence": round(confidence, 4),
#         "probabilities": predictions
#     }
    
#     return output

@router.post("/")
def predict(request: DRPredictionRequest):
    print("/predict endpoint calling")
    fundus_url = request.image_data.fundus
    oct_url = request.image_data.oct
    tabular_data = request.tabular_data
    
    results = {}
    results["combined_predictions"] = {}
    
    try:
        response = requests.get(fundus_url)
        response.raise_for_status()
        image_file = BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download image: {e}")

    fundus_image_preprocessed = preprocess_upload(image_file)
    predictions, predict_class, confidence = get_prediction(fundus_model, fundus_image_preprocessed)
    results["fundus"] = {
        "image_path": fundus_url,
        "prediction": CLASS_LABELS[predict_class],
        "class_index": predict_class,
        "confidence": round(confidence, 4),
        "probabilities": predictions
    }
    
    if oct_url:
        try:
            response = requests.get(oct_url)
            response.raise_for_status()
            image_file = BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Could not download image: {e}")
        
        oct_image_preprocessed = preprocess_oct(image_file)
        predictions, predict_class, confidence = get_prediction(oct_model, oct_image_preprocessed)
        results["oct"] = {
            "image_path": oct_url,
            "prediction": OCT_LABELS[predict_class],
            "class_index": predict_class,
            "confidence": round(confidence, 4),
            "probabilities": predictions
        }
        
    if tabular_data:
        data_preprocessed = preprocess_tabular(tabular_data)
        
        X = np.array([[data_preprocessed[f] for f in FEATURE_ORDER]])
    
        dr_prediction = tabular_model_dr.predict(X)[0]
        dme_prediction = tabular_model_dme.predict(X)[0]
        
        dr_prob = tabular_model_dr.predict_proba(X).tolist()
        dme_prob = tabular_model_dme.predict_proba(X).tolist()
        
        results["health_data"] = {
            "dr_prediction": int(dr_prediction),
            "dme_prediction": int(dme_prediction),
            "dr_probabilities": dr_prob[0], # check this [0]
            "dme_probabilities": dme_prob[0] # check this [0]
        }
    
    # fundus should be there for sure if success
    if "fundus" in results and "health_data" in results:
        results["combined_predictions"]["dr"] = fuse_dr_prediction(
            fundus_result=results["fundus"],
            tabular_result=results["health_data"],
            dm_time=data_preprocessed["dm_time"]
        )
    
    # if both oct and health data exits get combined
    if "health_data" in results and "oct" in results:
        results["combined_predictions"]["dme"] = fuse_dme_prediction(
            oct_result=results["oct"],
            tabular_result=results["health_data"]
        )
        
    if not results["combined_predictions"]:
        results.pop("combined_predictions")
        
    if not results:
        raise HTTPException(status_code=400, detail="No data provided for prediction")
        
    return results