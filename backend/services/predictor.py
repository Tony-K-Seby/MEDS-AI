import joblib
import numpy as np
import os

# Load models and metadata once at startup
MODELS_DIR = "ml_models"
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

symptom_index = metadata["symptom_index"]
prediction_classes = metadata["prediction_classes"]

def get_ensemble_prediction(models, X):
    """Get the ensemble prediction by averaging probabilities."""
    predictions = np.array([model.predict_proba(X) for model in models])
    avg_proba = np.mean(predictions, axis=0)
    return prediction_classes[np.argmax(avg_proba, axis=1)[0]]

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

    # Get ensemble prediction
    final_pred = get_ensemble_prediction([rf_model, nb_model, svm_model], input_data)

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


####################
############################################ensemble model############################################
###############

# import joblib
# import numpy as np
# import pandas as pd
# import os
# from typing import List, Dict, Any
# from scipy.stats import mode

# # Load models and preprocessing objects
# MODELS_DIR = "ml_models"
# voting_clf = joblib.load(os.path.join(MODELS_DIR, "voting_clf.pkl"))
# feature_selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.pkl"))
# scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
# label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
# metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

# # Get metadata
# selected_features = metadata["selected_features"]
# prediction_classes = metadata["prediction_classes"]
# symptom_index = {symptom: idx for idx, symptom in enumerate(selected_features)}

# def normalize_symptom(symptom: str) -> str:
#     """Normalize symptom string for consistent matching."""
#     return ' '.join(word.capitalize() for word in symptom.strip().split())

# def validate_symptoms(symptoms: List[str]) -> List[str]:
#     """Validate and normalize input symptoms."""
#     if not symptoms or not isinstance(symptoms, list):
#         raise ValueError("Symptoms must be provided as a non-empty list")
    
#     if len(symptoms) > 10:
#         raise ValueError("Maximum 10 symptoms allowed")
    
#     valid_symptoms = []
#     for symptom in symptoms:
#         norm_symptom = normalize_symptom(symptom)
#         if norm_symptom in symptom_index:
#             valid_symptoms.append(norm_symptom)
    
#     if not valid_symptoms:
#         raise ValueError("No valid symptoms provided")
    
#     return valid_symptoms

# def create_feature_vector(symptoms: List[str]) -> pd.DataFrame:
#     """Create a pandas DataFrame with proper feature names."""
#     # Initialize a zero-filled dictionary with all features
#     feature_dict = {feature: 0 for feature in selected_features}
    
#     # Set 1 for present symptoms
#     for symptom in symptoms:
#         if symptom in feature_dict:
#             feature_dict[symptom] = 1
    
#     # Create DataFrame with a single row
#     return pd.DataFrame([feature_dict])

# def predict_disease(input_json: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Handle disease prediction with improved model pipeline.
    
#     Args:
#         input_json: Dictionary containing symptoms list
    
#     Returns:
#         Dictionary containing predictions and confidence scores
#     """
#     try:
#         # Validate input symptoms
#         input_symptoms = validate_symptoms(input_json.get("symptoms", []))
        
#         # Create feature DataFrame with proper column names
#         input_data = create_feature_vector(input_symptoms)
        
#         # Apply feature selection and scaling
#         input_data_selected = feature_selector.transform(input_data)
#         input_data_scaled = scaler.transform(input_data_selected)
        
#         # Get predictions from voting classifier
#         final_prediction = voting_clf.predict(input_data_scaled)[0]
#         probabilities = voting_clf.predict_proba(input_data_scaled)[0]
        
#         # Get individual model predictions
#         predictions = {}
#         confidence_scores = {}
        
#         for idx, (name, model) in enumerate(voting_clf.named_estimators_.items()):
#             pred = model.predict(input_data_scaled)[0]
#             proba = model.predict_proba(input_data_scaled)[0]
            
#             model_name = name.lower()  # rf, nb, or svm
#             predictions[f"{model_name}_prediction"] = prediction_classes[pred]
#             confidence_scores[model_name] = float(max(proba))
        
#         return {
#             **predictions,
#             "final_prediction": prediction_classes[final_prediction],
#             "confidence_scores": confidence_scores,
#             "input_symptoms": input_symptoms
#         }
        
#     except ValueError as e:
#         raise ValueError(str(e))
#     except Exception as e:
#         raise Exception(f"Prediction failed: {str(e)}")
    
######################
########################################dl model############################################
######################

# import joblib
# import numpy as np
# import pandas as pd
# import os
# from typing import List, Dict, Any
# from scipy.stats import mode
# import tensorflow as tf

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get backend directory
# MODELS_DIR = os.path.join(BASE_DIR, "ml_models") 

# # Load models and preprocessing objects
# #MODELS_DIR = "ml_models"  # Changed from "ml_models" to match the directory in your new code
# metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
# feature_selector = joblib.load(os.path.join(MODELS_DIR, "feature_selector.pkl"))
# scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
# label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# # Load the appropriate model based on metadata
# best_model_type = metadata.get("best_model_type", "hybrid")  # Default to hybrid if not specified
# selected_features = metadata["selected_features"]
# prediction_classes = metadata["prediction_classes"]
# symptom_index = {symptom: idx for idx, symptom in enumerate(selected_features)}

# # Load the appropriate model
# if best_model_type == "hybrid":
#     # Load the deep learning hybrid model
#     hybrid_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "hybrid_model.keras"))
    
#     # Load the ML components of the hybrid model
#     rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
#     gb_model = joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl"))
#     svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    
#     ml_models = {
#         "rf": rf_model,
#         "gb": gb_model,
#         "svm": svm_model
#     }
# else:  # best_model_type == "dnn"
#     # Load only the DNN model
#     dnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dnn_model.keras"))

# def normalize_symptom(symptom: str) -> str:
#     """Normalize symptom string for consistent matching."""
#     return ' '.join(word.capitalize() for word in symptom.strip().split())

# def validate_symptoms(symptoms: List[str]) -> List[str]:
#     """Validate and normalize input symptoms."""
#     if not symptoms or not isinstance(symptoms, list):
#         raise ValueError("Symptoms must be provided as a non-empty list")
    
#     if len(symptoms) > 10:
#         raise ValueError("Maximum 10 symptoms allowed")
    
#     valid_symptoms = []
#     for symptom in symptoms:
#         norm_symptom = normalize_symptom(symptom)
#         if norm_symptom in symptom_index:
#             valid_symptoms.append(norm_symptom)
    
#     if not valid_symptoms:
#         raise ValueError("No valid symptoms provided")
    
#     return valid_symptoms

# def create_feature_vector(symptoms: List[str]) -> pd.DataFrame:
#     """Create a pandas DataFrame with proper feature names."""
#     # Initialize a zero-filled dictionary with all features
#     feature_dict = {feature: 0 for feature in selected_features}
    
#     # Set 1 for present symptoms
#     for symptom in symptoms:
#         if symptom in feature_dict:
#             feature_dict[symptom] = 1
    
#     # Create DataFrame with a single row
#     return pd.DataFrame([feature_dict])

# def predict_disease(input_json: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Handle disease prediction with the deep learning or hybrid model pipeline.
    
#     Args:
#         input_json: Dictionary containing symptoms list
    
#     Returns:
#         Dictionary containing predictions and confidence scores
#     """
#     try:
#         # Validate input symptoms
#         input_symptoms = validate_symptoms(input_json.get("symptoms", []))
        
#         # Create feature DataFrame with proper column names
#         input_data = create_feature_vector(input_symptoms)
        
#         # Apply feature selection and scaling
#         input_data_selected = feature_selector.transform(input_data)
#         input_data_scaled = scaler.transform(input_data_selected)
        
#         # Prediction logic depends on which model type was saved
#         if best_model_type == "hybrid":
#             # Get predictions from traditional ML models first
#             rf_preds = ml_models["rf"].predict_proba(input_data_scaled)
#             gb_preds = ml_models["gb"].predict_proba(input_data_scaled)
#             svm_preds = ml_models["svm"].predict_proba(input_data_scaled)
            
#             # Use the hybrid model for final prediction
#             hybrid_pred_probs = hybrid_model.predict(
#                 [input_data_scaled, rf_preds, gb_preds, svm_preds]
#             )[0]
            
#             final_prediction = np.argmax(hybrid_pred_probs)
            
#             # Create result dict with individual model predictions
#             result = {
#                 "rf_prediction": prediction_classes[np.argmax(rf_preds[0])],
#                 "gb_prediction": prediction_classes[np.argmax(gb_preds[0])],
#                 "svm_prediction": prediction_classes[np.argmax(svm_preds[0])],
#                 "final_prediction": prediction_classes[final_prediction],
#                 "confidence_scores": {
#                     "rf": float(np.max(rf_preds[0])),
#                     "gb": float(np.max(gb_preds[0])),
#                     "svm": float(np.max(svm_preds[0])),
#                     "hybrid": float(hybrid_pred_probs[final_prediction])
#                 },
#                 "input_symptoms": input_symptoms
#             }
            
#         else:  # DNN model
#             # Get prediction from DNN
#             dnn_pred_probs = dnn_model.predict(input_data_scaled)[0]
#             final_prediction = np.argmax(dnn_pred_probs)
            
#             result = {
#                 "final_prediction": prediction_classes[final_prediction],
#                 "confidence_scores": {
#                     "dnn": float(dnn_pred_probs[final_prediction])
#                 },
#                 "input_symptoms": input_symptoms
#             }
        
#         return result
        
#     except ValueError as e:
#         raise ValueError(str(e))
#     except Exception as e:
#         raise Exception(f"Prediction failed: {str(e)}")

# # Example usage
# if __name__ == "__main__":
#     # Test with sample symptoms
#     test_input = {
#         "symptoms": ["Itching", "Skin Rash", "Nodal Skin Eruptions"]
#     }
    
#     try:
#         result = predict_disease(test_input)
#         print("\nPrediction Result:")
#         print(f"Final Prediction: {result['final_prediction']}")
#         print("\nConfidence Scores:")
#         for model, score in result['confidence_scores'].items():
#             print(f"- {model.upper()}: {score:.4f}")
        
#         if "rf_prediction" in result:
#             print("\nIndividual Model Predictions:")
#             print(f"- Random Forest: {result['rf_prediction']}")
#             print(f"- Gradient Boosting: {result['gb_prediction']}")
#             print(f"- SVM: {result['svm_prediction']}")
        
#         print(f"\nInput Symptoms: {', '.join(result['input_symptoms'])}")
        
#     except Exception as e:
#         print(f"Error: {str(e)}")