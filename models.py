import keras
from keras.src.layers.core.dense import Dense

# --- MONKEY PATCH START ---
# This forces the Dense layer to ignore the 'quantization_config' argument
# This was done to fix an issue with tensorflo / keras version mismatch
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init
# --- MONKEY PATCH END ---

import joblib

FUNDUS_MODEL_PATH = 'models/fundus/best_efficientnet_b4.keras'
OCT_MODEL_PATH = 'models/oct/best_oct_model.keras'
# TABULAR_MODEL_DR_PATH = 'models/health_data/dr_model.joblib'
# TABULAR_MODEL_DME_PATH = 'models/health_data/dme_model.joblib'
TABULAR_MODEL_DME_PATH = 'models/health_data/xgb_edema_model.joblib'

# Load with native keras
fundus_model = keras.models.load_model(FUNDUS_MODEL_PATH, compile=False)
oct_model = keras.models.load_model(OCT_MODEL_PATH, compile=False)

# tabular_model_dr = joblib.load(TABULAR_MODEL_DR_PATH)
tabular_model_dme = joblib.load(TABULAR_MODEL_DME_PATH)