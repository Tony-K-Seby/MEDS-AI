from flask import Blueprint, request, jsonify
from bson import ObjectId
from datetime import datetime, timedelta
from database import mongo

doctor_bp = Blueprint("doctor", __name__)

def generate_time_slots(start, end):
    slots = []
    current = start
    while current < end:
        slots.append(current.strftime("%I:%M %p"))
        current += timedelta(minutes=30)
    return slots

# Function to format time strings correctly
def format_time(time_str):
    if ":" not in time_str:
        time_str = time_str[:-2] + ":00 " + time_str[-2:]  # Convert '9AM' to '9:00 AM'
    return datetime.strptime(time_str, "%I:%M %p")

# API to fetch available slots for a doctor
@doctor_bp.route("/<doctor_id>/slots", methods=["GET"])
def get_available_slots(doctor_id):
    doctor = mongo.db.doctors.find_one({"_id": int(doctor_id)})
    if not doctor:
        return jsonify({"error": "Doctor not found"}), 404

    booked_slots = set()
    appointments = mongo.db.appointments.find({"doctor_id": int(doctor_id)})
    for appt in appointments:
        booked_slots.add(appt["slot"])

    final_slots = {}
    days_mapping = {
        "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
        "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
    }

    for entry in doctor.get("availability", []):
        parts = entry.split(": ")
        if len(parts) != 2:
            continue  # Skip malformed entries
        days, time_range = parts
        start_time, end_time = time_range.split("-")

        start_dt = format_time(start_time.strip())
        end_dt = format_time(end_time.strip())

        for day_abbr in days.split("-"):
            day_name = days_mapping.get(day_abbr.strip(), "")
            if day_name:
                slots = generate_time_slots(start_dt, end_dt)
                available_slots = [slot for slot in slots if slot not in booked_slots]
                final_slots[day_name] = available_slots

    return jsonify({"available_slots": final_slots})
