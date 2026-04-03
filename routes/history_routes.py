from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone
from google.cloud.firestore import Query

from schemas.save_prediction_schema import SavePredictionRequest
from database.database import get_db

router = APIRouter(prefix="/history")
db = get_db()

@router.get("/home")
def home():
    return {"msg": "hello from /history"}

# Save
@router.post("/save-prediction", status_code=status.HTTP_201_CREATED)
async def save_prediction(data: SavePredictionRequest):
    # Extract the nested clinical results
    dr_data = data.prediction.combined_predictions.get("dr", {})
    dme_data = data.prediction.combined_predictions.get("dme", {})
    
    doc = {
        "patient_name": data.patient_name,
        "patient_id": data.patient_id,
        "doctor_id": data.doctor_id,
        "fundus_image_url": data.fundus_image_url,
        "oct_image_url": data.oct_image_url,
        "health_data": data.health_data, 

        "prediction": {
            "combined_predictions": {
                "dr": {
                    "final_stage": dr_data.get("final_stage"),
                    "confidence": dr_data.get("confidence"),
                    "interpretation": dr_data.get("interpretation")
                },
                "dme": {
                    "final_prediction": dme_data.get("final_prediction"),
                    "confidence": dme_data.get("confidence"),
                    "interpretation": dme_data.get("interpretation")
                } if dme_data else None
            }
        },
        "saved_at": datetime.now(timezone.utc)
    }

    try:
        _, doc_ref = await db.collection("predictions").add(doc)
        
        return {
            "message": "Prediction saved successfully",
            "id": doc_ref.id 
        }
    except Exception as e:
        print(f"Firestore Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Could not save to database."
        )

# Get predictions by doctor id
@router.get("/predictions")
async def get_predictions(doctor_id: str):
    try:
        predictions_query = (
            db.collection("predictions")
            .where("doctor_id", "==", doctor_id)
            .order_by("saved_at", direction=Query.DESCENDING)
        )

        docs = await predictions_query.get()

        results = []
        for doc in docs:
            prediction_data = doc.to_dict()
            prediction_data["id"] = doc.id
            results.append(prediction_data)

        return results

    except Exception as e:
        print(f"Firestore Fetch Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving data from database."
        )