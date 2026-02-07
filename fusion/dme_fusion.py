import numpy as np

OCT_WEIGHT = 0.8
TABULAR_WEIGHT = 0.2
HIGH_CONF_THRESHOLD = 0.7

def fuse_dme_prediction(oct_result: dict, tabular_result: dict | None):
    class_labels = {
        0: "DME",
        1: "NORMAL",
        2: "OTHER_DISEASE"
    }
    oct_probs = np.array(oct_result["probabilities"]) # index 0 = DME
    oct_dme_prob = float(oct_probs[0])

    if tabular_result is None:
        final_prob = oct_dme_prob
        weights_used = {"oct": 1.0, "health_data": 0.0}
    else:
        tabular_probs = np.array(tabular_result["dme_probabilities"])
        tabular_dme_prob = float(tabular_probs[1])  # index 1 = DME

        # Weighted fusion
        final_prob = OCT_WEIGHT * oct_dme_prob + TABULAR_WEIGHT * tabular_dme_prob
        weights_used = {"oct": OCT_WEIGHT, "health_data": TABULAR_WEIGHT}

    if final_prob >= 0.5:
        final_prediction = "DME"
        confidence = final_prob
    else:
        final_prediction = "No DME"
        confidence = 1 - final_prob

    # interpretation
    if oct_dme_prob >= HIGH_CONF_THRESHOLD:
        interpretation = "Strong OCT evidence of DME"
    elif final_prob >= 0.5:
        interpretation = "Moderate DME risk (image + health data)"
    else:
        interpretation = "No significant evidence of DME"

    return {
        "final_prediction": final_prediction,
        "confidence": round(confidence, 4),
        "weights_used": weights_used,
        "interpretation": interpretation
    }
