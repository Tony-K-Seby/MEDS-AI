import datetime
from bson import ObjectId
from flask import Blueprint, current_app, request, jsonify
from services.predictor import predict_disease
from services.chatbot_predictor import get_conversation_summary, process_message, start_chat  # New chatbot logic
from database import mongo


# Create a Blueprint for prediction-related routes
prediction_bp = Blueprint("prediction", __name__)

# @prediction_bp.route("/predict", methods=["POST"])
# def predict():
#     try:
#         input_json = request.get_json()
#         if not input_json:
#             return jsonify({"error": "No data provided"}), 400

#         user_id = input_json.get("user_id")
#         symptoms = input_json.get("symptoms", [])

#         # Call prediction function
#         prediction_response = predict_disease({"symptoms": symptoms})

#         # Debugging logs
#         print("Prediction Response:", prediction_response)

#         disease_name = prediction_response.get("final_prediction")
        
#         if not disease_name:
#             return jsonify({"error": "Prediction failed, no disease detected"}), 500

#         # Fetch disease details from the database
#         disease_doc = mongo.db.diseases.find_one({"name": disease_name})

#         if disease_doc:
#             prediction_response["disease_id"] = str(disease_doc["_id"])
#             prediction_response["specialty_id"] = str(disease_doc["specialty_id"])
#         else:
#             prediction_response["disease_id"] = None
#             prediction_response["specialty_id"] = None
#             prediction_response["id_error"] = f"Disease '{disease_name}' not found in database"

#         # Store prediction in database if user is authenticated
#         if user_id and disease_doc:
#             try:
#                 prediction_record = {
#                     "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
#                     "symptoms": symptoms,
#                     "disease_id": disease_doc["_id"],
#                     "disease_name": disease_name,
#                     "specialty_id": disease_doc["specialty_id"],
#                     "final_prediction": disease_name,  # ✅ Ensuring this is stored
#                     "confidence_scores": prediction_response.get("confidence_scores", {}),
#                     "rf_prediction": prediction_response.get("rf_prediction"),
#                     "nb_prediction": prediction_response.get("nb_prediction"),
#                     "svm_prediction": prediction_response.get("svm_prediction"),
#                     "created_at": datetime.datetime.utcnow()
#                 }

#                 result = mongo.db.previous_predictions.insert_one(prediction_record)
#                 if result.inserted_id:
#                     prediction_response["saved_to_history"] = True
#             except Exception as e:
#                 print("Database Insertion Error:", str(e))
#                 prediction_response["db_error"] = str(e)
#                 prediction_response["saved_to_history"] = False

#         return jsonify(prediction_response)

#     except Exception as e:
#         print("Server Error:", str(e))
#         return jsonify({"error": str(e)}), 500
        
#     except Exception as e:
#         current_app.logger.error(f"Prediction error: {str(e)}")
#         return jsonify({"error": str(e)}), 500
    
#################################################################################
#################################################################################
#################################################################################

@prediction_bp.route("/predict", methods=["POST"])
async def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "No data provided"}), 400

        user_id = input_json.get("user_id")
        symptoms = input_json.get("symptoms", [])

        print(f"Received symptoms: {symptoms}")

        # Common daily diseases mapping
        common_diseases_mapping = {
            "Cough": ["Common Cold", "Flu", "Bronchitis"],
            "High Fever": ["Flu", "Common Cold", "Viral Infection"],
            "Headache": ["Tension Headache", "Migraine", "Stress"],
            "Stomach Pain": ["Gastritis", "Food Poisoning", "Indigestion"],
            "Nausea": ["Gastritis", "Food Poisoning", "Viral Infection"],
            "Vomiting": ["Gastroenteritis", "Food Poisoning"],
            "Diarrhea": ["Food Poisoning", "Viral Gastroenteritis"],
            "Skin Rash": ["Allergic Reaction", "Eczema", "Dermatitis"],
            "Itching": ["Allergic Reaction", "Dry Skin"],
            "Fatigue": ["Stress", "Common Cold", "Viral Infection"],
            "Sore Throat": ["Viral Infection", "Strep Throat"],
            "Runny Nose": ["Common Cold", "Allergic Rhinitis"],
            "Joint Pain": ["Muscle Strain", "Mild Arthritis"],
            "Back Pain": ["Muscle Strain", "Poor Posture"],
            "Anxiety": ["Stress", "Generalized Anxiety"],
            "Mild Fever": ["Viral Infection", "Common Cold"]
        }

        # Determine diseases by checking each input symptom
        possible_diseases = set()
        unmatched_symptoms = []

        for symptom in symptoms:
            matched = False
            for key, diseases in common_diseases_mapping.items():
                if symptom.strip().lower() == key.lower():
                    possible_diseases.update(diseases)
                    matched = True
                    break
            
            if not matched:
                unmatched_symptoms.append(symptom)

        print(f"Rule-based possible diseases: {possible_diseases}")

        # If there are unmatched symptoms or no diseases found, call ML prediction
        if unmatched_symptoms or not possible_diseases:
            try:
                # Call external ML prediction function
                ml_prediction = predict_disease({"symptoms": symptoms})
                print(f"ML Prediction raw output: {ml_prediction}")

                # Ensure final prediction is always treated as a list
                ml_final_prediction = ml_prediction.get("final_prediction", ["Unknown Disease"])
                if isinstance(ml_final_prediction, str):  # Convert to list if it's a string
                    ml_final_prediction = [ml_final_prediction]

                print(f"ML final prediction list: {ml_final_prediction}")

                possible_diseases = set(ml_final_prediction)
                prediction_response = ml_prediction
            except Exception as e:
                print("ML Prediction Error:", str(e))
                possible_diseases = {"Unknown Disease"}
                prediction_response = {
                    "final_prediction": list(possible_diseases),
                    "error": "ML prediction failed"
                }
        else:
            # Use rule-based prediction
            prediction_response = {
                "final_prediction": list(possible_diseases),
                "prediction_type": "rule-based",
                "confidence_scores": {"rule_based": 0.7}
            }

        print(f"Final possible diseases before DB lookup: {possible_diseases}")

        # Fetch primary disease details
        disease_doc = list(mongo.db.diseases.find({"name": {"$in": list(possible_diseases)}}))
        
        print(f"Disease documents from DB: {disease_doc}")

        # Prepare response details
        if disease_doc:
            primary_disease = disease_doc[0]  # Take the first matching disease
            prediction_response["disease_id"] = str(primary_disease["_id"])
            prediction_response["specialty_id"] = str(primary_disease.get("specialty_id", "Unknown"))
        else:
            prediction_response["disease_id"] = None
            prediction_response["specialty_id"] = None
            prediction_response["id_error"] = f"Disease(s) {list(possible_diseases)} not found in database"
            print(f"ID error generated: {prediction_response['id_error']}")

       # Store prediction in database if user is authenticated
        if user_id and disease_doc:
            try:
                primary_disease = disease_doc[0]  # Ensure we take the first matching disease
                
                prediction_record = {
                    "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
                    "symptoms": symptoms,
                    "disease_id": primary_disease["_id"],  # Extract from first disease doc
                    "disease_name": primary_disease["name"],  # Get name properly
                    "specialty_id": primary_disease.get("specialty_id", None),  # Handle missing specialty_id safely
                    "final_prediction": list(possible_diseases),
                    "prediction_type": prediction_response.get("prediction_type", "ml"),
                    "confidence_scores": prediction_response.get("confidence_scores", {}),
                    "unmatched_symptoms": unmatched_symptoms,
                    "created_at": datetime.datetime.now()
                }

                result = mongo.db.previous_predictions.insert_one(prediction_record)  # ❌ Remove `await`
                if result.inserted_id:
                    prediction_response["saved_to_history"] = True
            except Exception as e:
                print("Database Insertion Error:", str(e))
                prediction_response["db_error"] = str(e)
                prediction_response["saved_to_history"] = False

        print(f"Final Response: {prediction_response}")

        return jsonify(prediction_response)

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": str(e)}), 500
    
@prediction_bp.route("/delete_prediction", methods=["DELETE"])
async def delete_prediction():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "No data provided"}), 400

        user_id = input_json.get("user_id")
        prediction_id = input_json.get("prediction_id")

        if not user_id or not prediction_id:
            return jsonify({"error": "user_id and prediction_id are required"}), 400

        print(f"Received delete request for prediction_id: {prediction_id} by user_id: {user_id}")

        # Validate ObjectId
        try:
            prediction_oid = ObjectId(prediction_id)
        except Exception:
            return jsonify({"error": "Invalid prediction_id format"}), 400

        # Find and delete the prediction
        result = mongo.db.previous_predictions.delete_one({
            "_id": prediction_oid,
            "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id
        })

        if result.deleted_count == 0:
            return jsonify({"error": "No matching prediction found"}), 404

        print(f"Prediction {prediction_id} deleted successfully")
        return jsonify({"message": "Prediction deleted successfully"})

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": str(e)}), 500



#################################################################################
#################################################################################
#################################################################################

 # New routes for chatbot   
@prediction_bp.route("/start_chat", methods=["POST"])
def start_chat_route():
    """Route to start a new chat session"""
    data = request.get_json()
    session_id = data.get("session_id")
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    response = start_chat(session_id)
    return jsonify(response)

@prediction_bp.route("/process_message", methods=["POST"])
def process_message_route():
    """Route to process a user message and generate a response"""
    data = request.get_json()
    session_id = data.get("session_id")
    user_message = data.get("user_message")
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    if not user_message:
        return jsonify({"error": "No user message provided"}), 400
    
    response = process_message(session_id, user_message)
    return jsonify(response)

@prediction_bp.route("/get_summary", methods=["POST"])
def get_summary_route():
    """Route to get a summary of the conversation and diagnosis"""
    data = request.get_json()
    session_id = data.get("session_id")
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    summary = get_conversation_summary(session_id)
    return jsonify(summary)



###########previous_predictions
from bson import ObjectId

# @prediction_bp.route("/previous-predictions", methods=["GET"])
# def fetch_previous_predictions():
#     try:
#         # Extract user ID (required parameter)
#         user_id = request.args.get("user_id")
        
#         # Validate user ID
#         if not user_id:
#             return jsonify({"error": "User ID is required"}), 400

#         try:
#             user_id = ObjectId(user_id)  # Convert user_id to ObjectId
#         except:
#             return jsonify({"error": "Invalid user ID format"}), 400  # Handle invalid IDs

#         # Prepare query to find predictions for the user
#         query = {"user_id": user_id}

#         # Fetch all predictions for the user
#         predictions_cursor = mongo.db.previous_predictions.find(query)
#         predictions = list(predictions_cursor)  # Convert cursor to list

#         # Process predictions for response
#         processed_predictions = []
#         for prediction in predictions:
#             processed_pred = {
#                 "_id": str(prediction["_id"]),
#                 "user_id": str(prediction["user_id"]),  # Convert back to string for JSON response
#                 "symptoms": prediction["symptoms"],
#                 "disease_id": prediction["disease_id"],
#                 "disease_name": prediction["disease_name"],
#                 "specialty_id": prediction["specialty_id"],
#                 "created_at": prediction["created_at"].isoformat(),
#                 "confidence_scores": {
#                     "rf": prediction.get("confidence_scores", {}).get("rf"),
#                     "nb": prediction.get("confidence_scores", {}).get("nb"),
#                     "svm": prediction.get("confidence_scores", {}).get("svm")
#                 },
#                 "rf_prediction": prediction.get("rf_prediction"),
#                 "nb_prediction": prediction.get("nb_prediction"),
#                 "svm_prediction": prediction.get("svm_prediction")
#             }
#             processed_predictions.append(processed_pred)

#         return jsonify({"predictions": processed_predictions})

#     except Exception as e:
#         current_app.logger.error(f"Error fetching previous predictions: {str(e)}")
#         return jsonify({
#             "error": "Failed to fetch previous predictions", 
#             "details": str(e)
#         }), 500

@prediction_bp.route("/previous-predictions", methods=["GET"])
def fetch_previous_predictions():
    try:
        # Extract user ID (required parameter)
        user_id = request.args.get("user_id")
        
        # Validate user ID
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        try:
            user_id = ObjectId(user_id)  # Convert user_id to ObjectId
        except:
            return jsonify({"error": "Invalid user ID format"}), 400  # Handle invalid IDs

        # Prepare query to find predictions for the user
        query = {"user_id": user_id}

        # Fetch all predictions for the user
        predictions_cursor = mongo.db.previous_predictions.find(query)
        predictions = list(predictions_cursor)  # Convert cursor to list

        # Process predictions for response
        processed_predictions = []
        for prediction in predictions:
            processed_pred = {
                "_id": str(prediction["_id"]),
                "user_id": str(prediction["user_id"]),
                "symptoms": prediction["symptoms"],
                "disease_id": prediction["disease_id"],
                "disease_name": prediction["disease_name"],
                "specialty_id": prediction["specialty_id"],
                "created_at": prediction["created_at"].isoformat(),
                "prediction_type": prediction.get("prediction_type", "ml")
            }
            
            # Handle different prediction structures
            final_prediction = prediction.get("final_prediction")
            if isinstance(final_prediction, list):
                processed_pred["final_prediction"] = final_prediction
            else:
                processed_pred["final_prediction"] = {
                    "rf_prediction": prediction.get("rf_prediction", final_prediction),
                    "nb_prediction": prediction.get("nb_prediction", final_prediction),
                    "svm_prediction": prediction.get("svm_prediction", final_prediction)
                }
            
            # Handle different confidence score structures
            confidence_scores = prediction.get("confidence_scores", {})
            processed_pred["confidence_scores"] = {
                "rf": confidence_scores.get("rf"),
                "nb": confidence_scores.get("nb"),
                "svm": confidence_scores.get("svm"),
                "rule_based": confidence_scores.get("rule_based")
            }
            
            # Include unmatched symptoms if available
            if "unmatched_symptoms" in prediction:
                processed_pred["unmatched_symptoms"] = prediction["unmatched_symptoms"]
                
            processed_predictions.append(processed_pred)

        return jsonify({"predictions": processed_predictions})

    except Exception as e:
        current_app.logger.error(f"Error fetching previous predictions: {str(e)}")
        return jsonify({
            "error": "Failed to fetch previous predictions", 
            "details": str(e)
        }), 500