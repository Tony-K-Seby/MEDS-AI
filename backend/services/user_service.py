from database import mongo

def recommend_doctors(disease_id):
    """Get doctors specializing in a given disease"""
    doctors = mongo.doctors.find({"specialty_id": disease_id})
    return [{"id": str(doc["_id"]), "name": doc["name"]} for doc in doctors]

def predict_disease(symptoms):
    """Mock function for disease prediction"""
    # TODO: Replace with ML model prediction logic
    return {"predicted_disease": "Flu", "confidence": 0.85}
