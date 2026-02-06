import tensorflow as tf
import joblib

FUNDUS_MODEL_PATH = 'models/fundus/best_efficientnet_b4.keras'
OCT_MODEL_PATH = 'models/oct/best_oct_model.keras'
TABULAR_MODEL_DR_PATH = 'models/health_data/dr_model.joblib'
TABULAR_MODEL_DME_PATH = 'models/health_data/dme_model.joblib'

fundus_model = tf.keras.models.load_model(FUNDUS_MODEL_PATH)
oct_model = tf.keras.models.load_model(OCT_MODEL_PATH)
tabular_model_dr = joblib.load(TABULAR_MODEL_DR_PATH)
tabular_model_dme = joblib.load(TABULAR_MODEL_DME_PATH)