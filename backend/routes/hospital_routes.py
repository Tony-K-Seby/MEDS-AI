from flask import Blueprint, jsonify
from services.hospital_service import get_all_hospitals  # Import service function

hospital_bp = Blueprint("hospital", __name__)

@hospital_bp.route("/", methods=["GET"])
def fetch_hospitals():
    hospitals = get_all_hospitals()  # Call service function
    return jsonify(hospitals), 200
