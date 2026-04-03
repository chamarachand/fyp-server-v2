# routes/dr_routes.py
from fastapi import APIRouter, HTTPException
import numpy as np
import os
import requests
from io import BytesIO

from models import fundus_model, oct_model, tabular_model_dme
from utils.fundus_preprocessing import preprocess_fundus
from utils.oct_preprocessing import preprocess_oct
from utils.tabular_preprocessing import preprocess_tabular
from fusion.dr_fusion import fuse_dr_prediction_rule_based
from fusion.dme_fusion import fuse_dme_prediction
from schemas.dr_schema import DRPredictionRequest
from utils.image_validation import is_garbage_image

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

def validate_image_from_url(image_url: str, image_type: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_file = BytesIO(response.content)

        is_invalid, label = is_garbage_image(image_file)
        image_file.seek(0)

        if is_invalid:
            return None, True, f"{image_type} invalid ({label})"

        return image_file, False, None

    except requests.exceptions.RequestException as e:
        return None, True, f"{image_type} download error ({str(e)})"
    

@router.get("/home")
def home():
    return {"msg": "hello from server"}


@router.post("/")
def predict(request: DRPredictionRequest):
    print("/predict endpoint calling")
    fundus_url = request.image_data.fundus
    oct_url = request.image_data.oct
    tabular_data = request.tabular_data
    
    results = {}
    results["combined_predictions"] = {}

    fundus_file = None
    oct_file = None

    # Fundus image download
    try:
        response = requests.get(fundus_url)
        response.raise_for_status()
        fundus_file = BytesIO(response.content)

        is_invalid, label = is_garbage_image(fundus_file)
        if is_invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid fundus image detected ({label})"
            )

        fundus_file.seek(0)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not download fundus image: {e}")

    # OCT check and download
    if oct_url:
        try:
            response = requests.get(oct_url)
            response.raise_for_status()
            oct_file = BytesIO(response.content)

            is_invalid, label = is_garbage_image(oct_file)
            if is_invalid:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid OCT image detected ({label})"
                )

            oct_file.seek(0)

        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Could not download OCT image: {e}")

    # Fundus image prediction
    fundus_image_preprocessed = preprocess_fundus(fundus_file)
    predictions, predict_class, confidence = get_prediction(fundus_model, fundus_image_preprocessed)

    results["fundus"] = {
        "image_path": fundus_url,
        "prediction": CLASS_LABELS[predict_class],
        "class_index": predict_class,
        "confidence": round(confidence, 4),
        "probabilities": predictions
    }

    # OCt prediction (if exists)
    if oct_file:
        oct_image_preprocessed = preprocess_oct(oct_file)
        predictions, predict_class, confidence = get_prediction(oct_model, oct_image_preprocessed)

        results["oct"] = {
            "image_path": oct_url,
            "prediction": OCT_LABELS[predict_class],
            "class_index": predict_class,
            "confidence": round(confidence, 4),
            "probabilities": predictions
        }

    # Extract tabular metadata
    patient_facts_dict = None
    if tabular_data:
        # Convert Pydantic model to dictionary for the rule-based logic
        patient_facts_dict = tabular_data.dict() if hasattr(tabular_data, "dict") else tabular_data
        print(patient_facts_dict)

        data_preprocessed = preprocess_tabular(tabular_data)
        X = np.array([[data_preprocessed[f] for f in FEATURE_ORDER]])
        dme_prob = tabular_model_dme.predict_proba(X).tolist()
        
        results["health_data"] = {
            "input_features": patient_facts_dict,
            "dme_probabilities": dme_prob[0]
        }

    # Rule based fusion with metadata
    
    # Fundus DR fusion
    if "fundus" in results:
        results["combined_predictions"]["dr"] = fuse_dr_prediction_rule_based(
            fundus_result=results["fundus"],
            patient_data=patient_facts_dict
        )

    if "oct" in results:
        results["combined_predictions"]["dme"] = fuse_dme_prediction(
            oct_result=results["oct"],
            tabular_result=results.get("health_data")
        )
        
    if not results["combined_predictions"]:
        results.pop("combined_predictions")
        
    if not results:
        raise HTTPException(status_code=400, detail="No data provided for prediction")
        
    return results