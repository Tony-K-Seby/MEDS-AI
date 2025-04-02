from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import jwt
import datetime
import os
from database import mongo
from flask_cors import CORS

auth_bp = Blueprint("auth", __name__)
CORS(auth_bp)

# Load secret key from environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")

def validate_email(email):
    """Basic email validation"""
    return "@" in email and "." in email.split("@")[1]

# Add this to your auth.py
@auth_bp.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Auth blueprint is working!"}), 200

@auth_bp.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input fields
        required_fields = ["name", "email", "password", "gender", "age", "address", "contact_no"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Validate email format
        if not validate_email(data["email"]):
            return jsonify({"error": "Invalid email format"}), 400

        # Validate password length
        if len(data["password"]) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400

        # Validate age
        try:
            age = int(data["age"])
            if age < 1 or age > 120:
                return jsonify({"error": "Invalid age"}), 400
        except ValueError:
            return jsonify({"error": "Age must be a number"}), 400

        # Check if the user already exists
        existing_user = mongo.db.users.find_one({"email": data["email"].lower()})
        if existing_user:
            return jsonify({"error": "Email already exists"}), 409

        # Hash the password
        hashed_password = generate_password_hash(data["password"], method='pbkdf2:sha256')

        # Create user document
        user_data = {
            "name": data["name"].strip(),
            "email": data["email"].lower().strip(),
            "password": hashed_password,
            "gender": data["gender"].lower().strip(),
            "age": age,
            "address": data["address"].strip(),
            "contact_no": data["contact_no"].strip(),
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }

        # Insert into MongoDB
        result = mongo.db.users.insert_one(user_data)
        
        return jsonify({
            "message": "User registered successfully!",
            "user_id": str(result.inserted_id)
        }), 201

    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "An error occurred during registration"}), 500

@auth_bp.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input fields
        email = data.get("email", "").lower().strip()
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Find user by email
        user = mongo.db.users.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        # Check password
        if not check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid email or password"}), 401

        # Generate JWT token
        token_payload = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        
        token = jwt.encode(
            token_payload,
            SECRET_KEY,
            algorithm="HS256"
        )

        return jsonify({
            "token": token,
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"]
            },
            "message": "Login successful!"
        }), 200

    except Exception as e:
        current_app.logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "An error occurred during login"}), 500

# Optional: Add a token verification route
@auth_bp.route("/verify-token", methods=["GET"])
def verify_token():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "No token provided"}), 401

        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        
        # Check if user still exists
        user = mongo.db.users.find_one({"_id": ObjectId(payload["user_id"])})
        if not user:
            return jsonify({"error": "User not found"}), 401

        return jsonify({
            "valid": True,
            "user": {
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user["name"]
            }
        }), 200

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401
    except Exception as e:
        current_app.logger.error(f"Token verification error: {str(e)}")
        return jsonify({"error": "An error occurred during token verification"}), 500