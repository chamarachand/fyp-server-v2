import numpy as np

IMAGE_WEIGHT = 0.80
TABULAR_WEIGHT = 0.20

import numpy as np

class_labels = {
    0: "No DR",
    1: "Mild Non-Proliferative",
    2: "Moderate Non-Proliferative",
    3: "Severe Non-Proliferative",
    4: "Proliferative DR"
}

def calculate_clinical_risk(patient_data: dict) -> tuple[str, int]:
    score = 0
    print("calculate_clinical_risk calling")
    
    # HbA1c - The strongest
    # We give this a high weight
    hba1c = patient_data.get("hba1c")
    print("hb1ac", hba1c)
    if hba1c is not None:
        if hba1c >= 9.0:
            score += 4  
        elif hba1c >= 7.0:
            score += 2  

    # Diabetes time
    dm_time = patient_data.get("dm_time", 0)
    if dm_time >= 10:
        score += 3
    elif dm_time >= 5:
        score += 1
        
    # Smoking
    if patient_data.get("smoking", False) in [True, "Yes", "yes", "Y"]:
        score += 1
        
    # Age 
    if patient_data.get("age", 0) >= 60:
        score += 1
        
    # Alcohol
    if patient_data.get("alcohol_consumption", False) in [True, "Yes", "yes", "Y"]:
        score += 1

    if score >= 5:
        return "High Risk", score
    elif score >= 2:
        return "Moderate Risk", score
    else:
        return "Low Risk", score
 
import numpy as np

def fuse_dr_prediction_rule_based(fundus_result: dict, patient_data: dict | None) -> dict:
    fundus_probs = np.array(fundus_result["probabilities"])
    pred_class = int(np.argmax(fundus_probs))
    base_confidence = float(np.max(fundus_probs))
    
    final_confidence = base_confidence
    rule_applied = "Standard Image Prediction"
    risk_category = "Unknown"
    interpretation = "Prediction based on retinal image analysis"

    if patient_data:
        risk_category, risk_score = calculate_clinical_risk(patient_data)
        
        # Create a dynamic modifier (0.0 to 1.0 scale based on risk_score out of 10)
        risk_intensity = min(risk_score / 10.0, 1.0)

        # Rule A: Image says No DR, but Risk is High
        if pred_class == 0 and risk_category == "High Risk":
            penalty = 0.05 + (risk_intensity * 0.20)
            final_confidence = max(0.1, base_confidence * (1 - penalty))
            rule_applied = "Rule A: Dynamic Adjustment"

            interpretation = (
                "Although the retinal image does not show signs of diabetic retinopathy, "
                "the patient’s clinical profile suggests a high risk of disease. "
            )

        # Rule B: Severe image findings in Low Risk patient
        elif pred_class >= 3 and risk_category == "Low Risk":
            penalty = 0.30 - (risk_intensity * 0.20)
            final_confidence = max(0.1, base_confidence * (1 - penalty))
            rule_applied = "Rule B: Clinical Verification"

            interpretation = (
                "The retinal image suggests a more advanced stage of diabetic retinopathy, "
                "however, the patient’s clinical profile indicates a low risk. "
                "This inconsistency has been considered, as it may indicate a possible imaging artifact "
            )

        # Rule C: Image says DR, and Risk is High
        elif pred_class >= 1 and risk_category == "High Risk":
            boost = 0.05 + (risk_intensity * 0.10)
            final_confidence = min(0.99, base_confidence * (1 + boost))
            rule_applied = "Rule C: Clinical Support"

            interpretation = (
                f"The image indicates {class_labels[pred_class]}, and this finding is supported by the patient’s "
                "clinical profile, which indicates a high risk of diabetic retinopathy. "
            )

        # Standard Alignment
        else:
            rule_applied = "Rule D: Aligned Assessment"

            interpretation = (
                f"The retinal image findings are consistent with the patient’s {risk_category.lower()} clinical profile. "
            )

    return {
        "final_stage": class_labels.get(pred_class, "Unknown"),
        "class_index": pred_class,
        "confidence": round(final_confidence, 4),
        "probabilities": fundus_probs.round(4).tolist(),
        "clinical_risk_score": risk_score if patient_data else 0,
        "risk_category": risk_category,
        "rule_applied": rule_applied,
        "interpretation": interpretation
    }