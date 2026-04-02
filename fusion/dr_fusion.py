import numpy as np

IMAGE_WEIGHT = 0.80
TABULAR_WEIGHT = 0.20

class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def fuse_dr_prediction(fundus_result: dict, tabular_result: dict | None, dm_time: float) -> dict:
    print(f"dm_time: {dm_time}")
    
    fundus_probs = np.array(fundus_result["probabilities"])
    interpretation = "Prediction based strictly on retinal image analysis."
    
    # No tabular data available
    if tabular_result is None:
        combined_probs = fundus_probs
        weights_used = {"fundus": 1.0, "health_data": 0.0, "strategy": "Fundus Only"}
        
    else:
        tabular_probs = np.array(tabular_result["dr_probabilities"])
        
        fundus_pred_class = np.argmax(fundus_probs)
        tabular_pred_class = np.argmax(tabular_probs)
        
        # Override Strategy: High DM duration supports image
        if dm_time >= 10 and fundus_pred_class >= 2 and tabular_pred_class == 0:
            combined_probs = fundus_probs
            weights_used = {
                "fundus": 1.0, 
                "health_data": 0.0, 
                "strategy": "Override: High DM Duration supports Image"
            }
            interpretation = (
                f"Clear signs of diabetic retinopathy found in image. "
                f"Given the {int(dm_time)}-year history of diabetes, the image-based "
                "finding overrides the tabular model."
            )
            
        # Conservative Strategy: Low DM duration doubts image
        elif dm_time < 5 and fundus_pred_class >= 3:
            combined_probs = (0.5 * fundus_probs) + (0.5 * tabular_probs)
            weights_used = {
                "fundus": 0.5, 
                "health_data": 0.5, 
                "strategy": "Conservative: Low DM Duration doubts Image"
            }
            interpretation = (
                f"Image suggests severe DR, which is uncommon for a patient with only "
                f"{int(dm_time)} years of diabetes. Confidence reduced; manual review recommended."
            )

        # 4. Standard weighted strategy
        else:
            combined_probs = (IMAGE_WEIGHT * fundus_probs) + (TABULAR_WEIGHT * tabular_probs)
            weights_used = {
                "fundus": IMAGE_WEIGHT, 
                "health_data": TABULAR_WEIGHT, 
                "strategy": "Standard Weighted"
            }
            
            if fundus_pred_class == tabular_pred_class:
                interpretation = (
                    f"Both retinal image and clinical data consistently indicate "
                    f"{class_labels[fundus_pred_class]}."
                )
            else:
                interpretation = (
                    f"Final prediction is based mainly on retinal image findings "
                    f"({class_labels[fundus_pred_class]}), while clinical data suggested "
                    f"{class_labels[tabular_pred_class]}."
                )

    combined_class = int(np.argmax(combined_probs))
    combined_confidence = float(np.max(combined_probs))
    
    return {
        "final_stage": class_labels.get(combined_class, "Unknown"),
        "class_index": combined_class,
        "confidence": round(combined_confidence, 4),
        "probabilities": combined_probs.round(4).tolist(),
        "weights_used": weights_used,
        "interpretation": interpretation
    }