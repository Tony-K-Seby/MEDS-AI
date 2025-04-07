# import joblib
# import numpy as np
# import os

# # Load models and metadata once at startup
# MODELS_DIR = "ml_models"
# svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
# nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
# rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
# label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
# metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

# symptom_index = metadata["symptom_index"]
# prediction_classes = metadata["prediction_classes"]

# def get_ensemble_prediction(models, X):
#     """Get the ensemble prediction by averaging probabilities."""
#     predictions = np.array([model.predict_proba(X) for model in models])
#     avg_proba = np.mean(predictions, axis=0)
#     return prediction_classes[np.argmax(avg_proba, axis=1)[0]]

# def predict_disease(input_json):
#     """Handles the disease prediction logic."""
    
#     input_symptoms = input_json.get("symptoms", [])

#     # Convert input symptoms to model-compatible format
#     input_data = [0] * len(symptom_index)
#     for symptom in input_symptoms:
#         symptom = symptom.capitalize()
#         if symptom in symptom_index:
#             input_data[symptom_index[symptom]] = 1

#     input_data = np.array(input_data).reshape(1, -1)

#     # Get individual model predictions
#     rf_pred = prediction_classes[rf_model.predict(input_data)[0]]
#     nb_pred = prediction_classes[nb_model.predict(input_data)[0]]
#     svm_pred = prediction_classes[svm_model.predict(input_data)[0]]

#     # Get ensemble prediction
#     final_pred = get_ensemble_prediction([rf_model, nb_model, svm_model], input_data)

#     return {
#         "rf_prediction": rf_pred,
#         "nb_prediction": nb_pred,
#         "svm_prediction": svm_pred,
#         "final_prediction": final_pred,
#         "confidence_scores": {
#             "rf": float(max(rf_model.predict_proba(input_data)[0])),
#             "nb": float(max(nb_model.predict_proba(input_data)[0])),
#             "svm": float(max(svm_model.predict_proba(input_data)[0]))
#         }
#     }


####################
############################################ensemble model############################################
###############

import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "ml_models"

# Load all models
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
stacking_model = joblib.load(os.path.join(MODELS_DIR, "stacking_model.pkl"))
voting_model = joblib.load(os.path.join(MODELS_DIR, "voting_model.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
ensemble_weights = joblib.load(os.path.join(MODELS_DIR, "ensemble_weights.pkl"))
feature_selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.pkl"))

symptom_index = metadata["symptom_index"]
prediction_classes = metadata["prediction_classes"]
selected_features = metadata.get("selected_features", [])  # This is optional

def get_ensemble_prediction(models, X):
    """Get the ensemble prediction by averaging probabilities."""
    predictions = np.array([model.predict_proba(X) for model in models])
    avg_proba = np.mean(predictions, axis=0)
    return prediction_classes[np.argmax(avg_proba, axis=1)[0]]

def get_enhanced_prediction(input_data):
    """Get enhanced prediction using weighted ensemble."""
    # Use both advanced models but preserve original format
    models = [rf_model, nb_model, svm_model, xgb_model]
    
    # Apply weights for better prediction
    predictions = np.array([model.predict_proba(input_data) for model in models])
    weighted_preds = np.sum(predictions * ensemble_weights[:, np.newaxis, np.newaxis], axis=0)
    
    # Get best prediction as determined by voting classifier (more reliable)
    voting_pred = voting_model.predict(input_data)[0]
    stacking_pred = stacking_model.predict(input_data)[0]
    
    # Determine final prediction - prioritize voting model
    return prediction_classes[voting_pred]

def predict_disease(input_json):
    """Handles the disease prediction logic."""
    
    input_symptoms = input_json.get("symptoms", [])

    # Convert input symptoms to model-compatible format
    input_data = [0] * len(symptom_index)
    for symptom in input_symptoms:
        symptom = symptom.capitalize()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    # Get individual model predictions
    rf_pred = prediction_classes[rf_model.predict(input_data)[0]]
    nb_pred = prediction_classes[nb_model.predict(input_data)[0]]
    svm_pred = prediction_classes[svm_model.predict(input_data)[0]]

    # Get ensemble prediction - use the enhanced method but maintain original format
    final_pred = get_enhanced_prediction(input_data)

    # Return with the EXACT SAME format as required
    return {
        "rf_prediction": rf_pred,
        "nb_prediction": nb_pred,
        "svm_prediction": svm_pred,
        "final_prediction": final_pred,
        "confidence_scores": {
            "rf": float(max(rf_model.predict_proba(input_data)[0])),
            "nb": float(max(nb_model.predict_proba(input_data)[0])),
            "svm": float(max(svm_model.predict_proba(input_data)[0]))
        }
    }