import numpy as np

IMAGE_WEIGHT = 0.80
TABULAR_WEIGHT = 0.20

def fuse_dr_prediction(fundus_result: dict, tabular_result: dict | None, dm_time: int):
    print("dm_time", dm_time)
    class_labels = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }
    
    fundus_probs = np.array(fundus_result["probabilities"])
    interpretation = "Prediction based on retinal image analysis"
    
    if tabular_result is None:
        combined_probs = fundus_probs
        weights_used = {"fundus": 1.0, "health_data": 0.0}
    else:
        tabular_probs = np.array(tabular_result["dr_probabilities"])
        
        fundus_pred_class = np.argmax(fundus_probs)
        tabular_pred_class = np.argmax(tabular_probs)
        
        # Correction mechanisms (with dynamic weights)
        if dm_time >= 10 and fundus_pred_class >= 2 and tabular_pred_class == 0:
            combined_probs = fundus_probs
            weights_used = {
                "fundus": 1.0, 
                "health_data": 0.0, 
                "strategy": "Override: High DM Duration supports Image"
            }
            
            interpretation = (
                f"Clear signs of diabetic retinopathy were found in the retinal image. "
                f"Given the long history of diabetes ({int(dm_time)} years), the image-based "
                "finding is considered reliable."
            )
            
        elif dm_time < 5 and fundus_pred_class >= 3:
            combined_probs = (0.5 * fundus_probs + 0.5 * tabular_probs)
            weights_used = {
                "fundus": 0.5, 
                "health_data": 0.5, 
                "strategy": "Conservative: Low DM Duration doubts Image"
            }
            
            interpretation = (
                f"The retinal image suggests severe diabetic retinopathy, which is uncommon "
                f"for a patient with only {int(dm_time)} years of diabetes. "
                "The confidence has been reduced and manual review is recommended."
            )

        else:
            combined_probs = (IMAGE_WEIGHT * fundus_probs + TABULAR_WEIGHT * tabular_probs)
            weights_used = {
                "fundus": IMAGE_WEIGHT, 
                "health_data": TABULAR_WEIGHT, 
                "strategy": "Standard Weighted"
            }
            
            if fundus_pred_class == tabular_pred_class:
                interpretation = (
                    f"Both the retinal image and clinical data consistently indicate "
                    f"{class_labels[fundus_pred_class]}."
                )
            else:
                interpretation = (
                    f"The final prediction is based mainly on retinal image findings "
                    f"({class_labels[fundus_pred_class]}), while clinical data suggests "
                    f"{class_labels[tabular_pred_class]}."
                )

        
    combined_class = int(np.argmax(combined_probs))
    combined_confidence = float(np.max(combined_probs))
    
    return {
        "final_stage": class_labels.get(combined_class, str(combined_class))
                        if class_labels else combined_class,
        "class_index": combined_class,
        "confidence": round(combined_confidence, 4),
        "probabilities": combined_probs.round(4).tolist(),
        "weights_used": weights_used,
        "interpretation": interpretation
    }
