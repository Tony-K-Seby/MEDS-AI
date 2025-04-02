from bson import ObjectId
from database import mongo


def get_doctors_by_specialty(specialty_id):
    """Get all doctors for a specific specialty ID"""
    try:
        # Use specialty_id as integer
        specialty_id = int(specialty_id)
        doctors = list(mongo.db.doctors.find({"specialty_id": specialty_id}))
        
        # Convert ObjectIds to strings for JSON serialization
        for doctor in doctors:
            if isinstance(doctor["_id"], ObjectId):
                doctor["_id"] = str(doctor["_id"])
                
        return doctors
    except Exception as e:
        print(f"Error fetching doctors by specialty: {str(e)}")
        return []