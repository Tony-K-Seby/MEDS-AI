from datetime import datetime
from bson import ObjectId
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from services.user_service import recommend_doctors, predict_disease
from database import mongo

user_bp = Blueprint("user", __name__)

@user_bp.route("/recommend-doctors/<disease_id>", methods=["GET"])
def recommend_doctors_api(disease_id):
    doctors = recommend_doctors(disease_id)
    return jsonify(doctors), 200

@user_bp.route("/predict-disease", methods=["POST"])
def predict_disease_api():
    data = request.json
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    prediction = predict_disease(symptoms)
    return jsonify(prediction), 200

@user_bp.route("/doctors/specialty/<int:specialty_id>", methods=["GET"])
def get_doctors_by_specialty(specialty_id):
    try:
        # Find doctors with matching specialty_id (as integer)
        doctors = list(mongo.db.doctors.find({"specialty_id": specialty_id}))
        
        # Convert ObjectId to string for JSON serialization
        for doctor in doctors:
            if isinstance(doctor["_id"], ObjectId):
                doctor["_id"] = str(doctor["_id"])
        
        return jsonify({"doctors": doctors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@user_bp.route("/doctors", methods=["GET"])
def get_all_doctors():
    """ Retrieves all doctors with specialization, hospital details, and extended fields """
    doctors = mongo.db.doctors.aggregate([
        {
            "$lookup": {
                "from": "specializations",
                "localField": "specialty_id",
                "foreignField": "_id",  # Changed from specialty_id to _id
                "as": "specialization"
            }
        },
        {
            "$lookup": {
                "from": "hospitals",
                "localField": "hospital_id",
                "foreignField": "_id",  # Changed from hospital_id to _id
                "as": "hospital"
            }
        },
        {"$unwind": {"path": "$specialization", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$hospital", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "email": 1,
                "contact_no": 1,
                "education": 1,
                "experience": 1,
                "languages": 1,
                "procedures": 1,
                "availability": 1,
                "awards": 1,
                "address": 1,
                "specialization_name": {"$ifNull": ["$specialization.specialty_name", "Unknown"]},
                "hospital_name": {"$ifNull": ["$hospital.name", "Unknown"]},
                "hospital_address": {"$ifNull": ["$hospital.address", "Unknown"]},
                "hospital_city": {"$ifNull": ["$hospital.city", "Unknown"]},
                "hospital_state": {"$ifNull": ["$hospital.state", "Unknown"]},
                "hospital_pin_code": {"$ifNull": ["$hospital.pin_code", "Unknown"]},
                "hospital_contact_no": {"$ifNull": ["$hospital.contact_no", "Unknown"]}
            }
        }
    ])

    return jsonify({"doctors": list(doctors)})


@user_bp.route("/doctors/<int:doctor_id>", methods=["GET"])
def get_doctor_by_id(doctor_id):
    """ Retrieves a single doctor by ID with specialization & hospital details """
    doctor = mongo.db.doctors.aggregate([
        {"$match": {"_id": doctor_id}},  # Match integer ID
        {
            "$lookup": {
                "from": "specializations",
                "localField": "specialty_id",
                "foreignField": "_id",  # Changed from specialty_id to _id
                "as": "specialization"
            }
        },
        {
            "$lookup": {
                "from": "hospitals",
                "localField": "hospital_id",
                "foreignField": "_id",  # Changed from hospital_id to _id
                "as": "hospital"
            }
        },
        {"$unwind": {"path": "$specialization", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$hospital", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "email": 1,
                "contact_no": 1,
                "specialization_name": {"$ifNull": ["$specialization.specialty_name", "Unknown"]},
                "hospital_name": {"$ifNull": ["$hospital.name", "Unknown"]},
                "hospital_address": {"$ifNull": ["$hospital.address", "Unknown"]},
                "hospital_city": {"$ifNull": ["$hospital.city", "Unknown"]},
                "hospital_state": {"$ifNull": ["$hospital.state", "Unknown"]},
                "hospital_pin_code": {"$ifNull": ["$hospital.pin_code", "Unknown"]},
                "hospital_contact_no": {"$ifNull": ["$hospital.contact_no", "Unknown"]}
            }
        }
    ])

    doctor_data = list(doctor)

    if not doctor_data:
        return jsonify({"error": "Doctor not found"}), 404

    return jsonify({"doctor": doctor_data[0]})

@user_bp.route("/test-doctors", methods=["GET"])
def test_doctors():
    doctors = list(mongo.db.doctors.find({}))
    for doc in doctors:
        doc["_id"] = str(doc["_id"])
    return jsonify(doctors)


@user_bp.route("/disease/precautions", methods=["GET"])
def get_disease_precautions():
    try:
        # Get disease name from query parameter
        disease_name = request.args.get('disease')
        
        # Validate input
        if not disease_name:
            return jsonify({"error": "Disease name is required"}), 400
        
        # Find the disease in the database
        disease = mongo.db.precautions.find_one({"disease": disease_name})
        
        # Check if disease exists
        if not disease:
            return jsonify({
                "error": f"No precautions found for disease: {disease_name}",
                "disease": disease_name,
                "precautions": []
            }), 404
        
        # Extract precautions
        precautions = disease.get('precautions', [])
        
        # Prepare response
        response = {
            "disease": disease_name,
            "precautions": precautions
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Log the error for server-side tracking
        print(f"Error retrieving precautions: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred while fetching precautions",
            "details": str(e)
        }), 500
    



@user_bp.route("/book-appointment", methods=["POST"])
def book_appointment():
    try:
        data = request.json
        print("DEBUG - Received data:", data)  # Debug line to check received data
        
        user_id = data.get("user_id")
        doctor_id = data.get("doctor_id")
        selected_day = data.get("day")
        selected_slot = data.get("slot")

        # Validate all required fields
        if not user_id or not doctor_id or not selected_slot or not selected_day:
            print(f"DEBUG - Missing fields: user_id={user_id}, doctor_id={doctor_id}, day={selected_day}, slot={selected_slot}")
            return jsonify({"error": "Missing required fields"}), 400

        try:
            # Convert user_id to ObjectId
            user_id = ObjectId(user_id)
            
            # Handle doctor_id - if it's a number, keep it as is
            if isinstance(doctor_id, int) or (isinstance(doctor_id, str) and doctor_id.isdigit()):
                doctor_id = int(doctor_id)
            else:
                # Otherwise try to convert to ObjectId
                try:
                    doctor_id = ObjectId(doctor_id)
                except Exception as e:
                    print(f"DEBUG - Invalid doctor ID format: {str(e)}")
                    return jsonify({"error": "Invalid doctor ID format"}), 400
                    
        except Exception as e:
            print(f"DEBUG - Invalid user ID format: {str(e)}")
            return jsonify({"error": "Invalid user ID format"}), 400

        print(f"DEBUG - Processed IDs: user_id={user_id}, doctor_id={doctor_id}")

        # Check if the slot is already booked for the specific day
        existing_booking = mongo.db.appointments.find_one({
            "doctor_id": doctor_id,
            "day": selected_day,
            "slot": selected_slot
        })
        
        if existing_booking:
            print(f"DEBUG - Slot already booked: day={selected_day}, slot={selected_slot}")
            return jsonify({"error": "Slot already booked"}), 409

        # Save appointment in DB
        appointment = {
            "user_id": user_id,
            "doctor_id": doctor_id,
            "day": selected_day,
            "slot": selected_slot,
            "booked_at": datetime.now()
        }
        result = mongo.db.appointments.insert_one(appointment)
        print(f"DEBUG - Appointment created with ID: {result.inserted_id}")

        return jsonify({
            "message": "Appointment booked successfully",
            "appointment": {
                "user_id": str(user_id),
                "doctor_id": str(doctor_id) if isinstance(doctor_id, ObjectId) else doctor_id,
                "day": selected_day,
                "slot": selected_slot,
                "booked_at": str(appointment["booked_at"])  # Convert datetime to string
            }
        }), 201
        
    except Exception as e:
        print(f"DEBUG - Unhandled exception: {str(e)}")
        return jsonify({"error": "Server error", "message": str(e)}), 500
    


    # Get all appointments for a user
@user_bp.route("/appointments/<user_id>", methods=["GET"])
def get_user_appointments(user_id):
    try:
        # Convert user_id to ObjectId
        try:
            user_id = ObjectId(user_id)
        except Exception as e:
            print(f"DEBUG - Invalid user ID format: {str(e)}")
            return jsonify({"error": "Invalid user ID format"}), 400
        
        print(f"DEBUG - Fetching appointments for user ID: {user_id}")
        
        # Fetch appointments for this user
        appointments = list(mongo.db.appointments.find({"user_id": user_id}))
        
        print(f"DEBUG - Found {len(appointments)} appointments")
        
        # Convert ObjectId to string for JSON serialization
        for appointment in appointments:
            appointment["_id"] = str(appointment["_id"])
            appointment["user_id"] = str(appointment["user_id"])
            
            # Convert datetime to string if present
            if isinstance(appointment.get("booked_at"), datetime):
                appointment["booked_at"] = appointment["booked_at"].isoformat()
        
        return jsonify({"appointments": appointments}), 200
        
    except Exception as e:
        print(f"DEBUG - Error fetching appointments: {str(e)}")
        return jsonify({"error": "Server error", "message": str(e)}), 500

# Cancel an appointment
@user_bp.route("/cancel-appointment/<appointment_id>", methods=["DELETE"])
def cancel_appointment(appointment_id):
    try:
        # Convert appointment_id to ObjectId
        try:
            appointment_id = ObjectId(appointment_id)
        except Exception as e:
            print(f"DEBUG - Invalid appointment ID format: {str(e)}")
            return jsonify({"error": "Invalid appointment ID format"}), 400
            
        print(f"DEBUG - Attempting to cancel appointment ID: {appointment_id}")
            
        # Check if appointment exists
        appointment = mongo.db.appointments.find_one({"_id": appointment_id})
        if not appointment:
            print(f"DEBUG - Appointment not found: {appointment_id}")
            return jsonify({"error": "Appointment not found"}), 404
            
        # Delete the appointment
        result = mongo.db.appointments.delete_one({"_id": appointment_id})
        
        if result.deleted_count == 1:
            print(f"DEBUG - Successfully canceled appointment: {appointment_id}")
            return jsonify({"message": "Appointment cancelled successfully"}), 200
        else:
            print(f"DEBUG - Failed to cancel appointment: {appointment_id}")
            return jsonify({"error": "Failed to cancel appointment"}), 500
            
    except Exception as e:
        print(f"DEBUG - Error cancelling appointment: {str(e)}")
        return jsonify({"error": "Server error", "message": str(e)}), 500