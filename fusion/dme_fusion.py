# import numpy as np

# OCT_WEIGHT = 0.8
# TABULAR_WEIGHT = 0.2
# HIGH_CONF_THRESHOLD = 0.7

# def fuse_dme_prediction(oct_result: dict, tabular_result: dict | None):
#     oct_probs = np.array(oct_result["probabilities"]) # index 0 = DME
#     oct_dme_prob = float(oct_probs[0])

#     if tabular_result is None:
#         final_prob = oct_dme_prob
#         weights_used = {"oct": 1.0, "health_data": 0.0}
#     else:
#         tabular_probs = np.array(tabular_result["dme_probabilities"])
#         tabular_dme_prob = float(tabular_probs[1])  # index 1 = DME

#         # Weighted fusion
#         final_prob = OCT_WEIGHT * oct_dme_prob + TABULAR_WEIGHT * tabular_dme_prob
#         weights_used = {"oct": OCT_WEIGHT, "health_data": TABULAR_WEIGHT}

#     if final_prob >= 0.5:
#         final_prediction = "DME"
#         confidence = final_prob
#     else:
#         final_prediction = "No DME"
#         confidence = 1 - final_prob

#     # interpretation
#     if oct_dme_prob >= HIGH_CONF_THRESHOLD:
#         interpretation = "Strong OCT evidence of DME"
#     elif final_prob >= 0.5:
#         interpretation = "Moderate DME risk (image + health data)"
#     else:
#         interpretation = "No significant evidence of DME"

#     return {
#         "final_prediction": final_prediction,
#         "confidence": round(confidence, 4),
#         "weights_used": weights_used,
#         "interpretation": interpretation
#     }

import numpy as np

OCT_WEIGHT = 0.8
TABULAR_WEIGHT = 0.2
HIGH_CONF_THRESHOLD = 0.7
TABULAR_OPTIMAL_THRESH = 0.3670  # The tuned F2 threshold from XGBoost

def calibrate_probability(raw_prob, threshold):
    """
    Scales the probability so that the optimal threshold maps exactly to 0.5.
    This ensures it plays nicely in a standard weighted average.
    """
    if raw_prob < threshold:
        # Scale 0 to threshold -> 0 to 0.5
        return 0.5 * (raw_prob / threshold)
    else:
        # Scale threshold to 1.0 -> 0.5 to 1.0
        return 0.5 + 0.5 * ((raw_prob - threshold) / (1.0 - threshold))

def fuse_dme_prediction(oct_result: dict, tabular_result: dict | None):
    # OCT probabilities (assuming index 0 is DME)
    oct_probs = np.array(oct_result["probabilities"]) 
    oct_dme_prob = float(oct_probs[0])

    if tabular_result is None:
        final_prob = oct_dme_prob
        weights_used = {"oct": 1.0, "health_data": 0.0}
    else:
        # Tabular probabilities (index 1 is DME)
        tabular_probs = np.array(tabular_result["dme_probabilities"])
        raw_tabular_prob = float(tabular_probs[1]) 
        
        # Calibrate the tabular probability before fusing!
        calibrated_tabular_prob = calibrate_probability(raw_tabular_prob, TABULAR_OPTIMAL_THRESH)

        # Weighted fusion with calibrated data
        final_prob = (OCT_WEIGHT * oct_dme_prob) + (TABULAR_WEIGHT * calibrated_tabular_prob)
        weights_used = {"oct": OCT_WEIGHT, "health_data": TABULAR_WEIGHT}

    # Final Classification
    if final_prob >= 0.5:
        final_prediction = "DME"
        confidence = final_prob
    else:
        final_prediction = "No DME"
        confidence = 1 - final_prob

    # Interpretation Logic
    if oct_dme_prob >= HIGH_CONF_THRESHOLD:
        interpretation = "Strong OCT evidence of DME"
    elif final_prob >= 0.5:
        if oct_dme_prob < 0.5 and raw_tabular_prob >= TABULAR_OPTIMAL_THRESH:
            interpretation = "Moderate DME risk (Elevated by patient health history)"
        else:
            interpretation = "Moderate DME risk (Combined image + health data)"
    else:
        interpretation = "No significant evidence of DME"

    return {
        "final_prediction": final_prediction,
        "confidence": round(confidence, 4),
        "weights_used": weights_used,
        "interpretation": interpretation
    }