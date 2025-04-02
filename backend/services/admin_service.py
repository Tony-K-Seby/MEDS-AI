from database import mongo

def add_doctor(doctor_data):
    """Add a new doctor"""
    mongo.doctors.insert_one(doctor_data)
    return {"message": "Doctor added successfully!"}

def get_all_doctors():
    """Retrieve all doctors"""
    doctors = mongo.doctors.find()
    return [{"id": str(doc["_id"]), "name": doc["name"], "specialty_id": doc["specialty_id"]} for doc in doctors]

def get_doctor_by_id(doctor_id):
    """Retrieve doctor by ID"""
    doctor = mongo.doctors.find_one({"_id": doctor_id})
    if doctor:
        return {"id": str(doctor["_id"]), "name": doctor["name"], "specialty_id": doctor["specialty_id"]}
    return {"error": "Doctor not found"}
