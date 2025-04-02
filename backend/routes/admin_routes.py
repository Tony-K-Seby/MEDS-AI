from flask import Blueprint, request, jsonify
from services.admin_service import add_doctor, get_all_doctors, get_doctor_by_id

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/doctors", methods=["POST"])
def add_doctor_api():
    doctor_data = request.json
    result = add_doctor(doctor_data)
    return jsonify(result), 201

@admin_bp.route("/doctors", methods=["GET"])
def get_all_doctors_api():
    return jsonify(get_all_doctors()), 200

@admin_bp.route("/doctors/<doctor_id>", methods=["GET"])
def get_doctor_by_id_api(doctor_id):
    return jsonify(get_doctor_by_id(doctor_id)), 200
