from database import mongo

def get_all_hospitals():
    hospitals = list(mongo.db.hospitals.find({}, {"_id": 0}))  # Excluding MongoDB _id for cleaner JSON
    return hospitals
