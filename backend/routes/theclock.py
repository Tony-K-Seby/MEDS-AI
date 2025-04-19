from flask import Blueprint, request, jsonify, current_app
import datetime
import random

from flask_cors import cross_origin
from database import mongo  # Import the already initialized mongo from your 'database' file

theclock_bp = Blueprint('theclock', __name__)

@theclock_bp.route("/", methods=["POST", "GET", "OPTIONS"])
@cross_origin(origins=["*"], allow_headers=["Content-Type", "Authorization", "Accept"],
              methods=["POST", "GET", "OPTIONS"], supports_credentials=False)
def the_clock():
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return "", 200
        
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            user_email = data.get("email")
            if not user_email:
                return jsonify({"error": "Email is required"}), 400
            
            # Check if the user exists
            user = mongo.db.users.find_one({"email": user_email.lower().strip()})
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # Check if the death date already exists
            existing_entry = mongo.db.theclock.find_one({"user_email": user_email.lower().strip()})
            if existing_entry:
                return jsonify({
                    "message": "Your death date was already determined.",
                    "death_date": existing_entry["death_date"],
                    "countdown_started": existing_entry["created_at"]
                }), 200
            
            # Collect form parameters
            dob_day = data.get("dob_day")
            dob_month = data.get("dob_month")
            dob_year = data.get("dob_year")
            sex = data.get("sex")
            smoker = data.get("smoker", False)
            bmi = data.get("bmi")
            outlook = data.get("outlook")
            alcohol = data.get("alcohol")
            country = data.get("country")
            
            # Fitness and diet parameters
            include_fitness_diet = data.get("include_fitness_diet", False)
            fitness_level = data.get("fitness_level") if include_fitness_diet else None
            diet_rating = data.get("diet_rating") if include_fitness_diet else None
            
            # Create DOB datetime object
            try:
                dob = datetime.datetime(int(dob_year), int(dob_month), int(dob_day))
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid date of birth"}), 400
            
            # Calculate current age
            today = datetime.datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Base life expectancy logic
            life_expectancy = 80
            
            # Sex adjustment
            if sex.lower() == "male":
                life_expectancy -= 5
            
            # Smoking adjustment
            if smoker:
                life_expectancy -= 10
            
            # BMI adjustment
            if bmi == "<25":
                life_expectancy += 2
            elif bmi == "25-30":
                life_expectancy -= 2
            elif bmi == ">30":
                life_expectancy -= 8
            
            # Outlook adjustment
            if outlook == "Positive":
                life_expectancy += 5
            elif outlook == "Negative":
                life_expectancy -= 5
            
            # Alcohol adjustment
            if alcohol == "Heavy":
                life_expectancy -= 8
            elif alcohol == "Moderate":
                life_expectancy -= 2
            elif alcohol == "None":
                life_expectancy += 3
            
            # Country-based adjustment (simplified)
            if country in ["Japan", "Switzerland", "Singapore", "Australia", "Sweden"]:
                life_expectancy += 4
            elif country in ["Sierra Leone", "Central African Republic", "Chad"]:
                life_expectancy -= 5
            
            # Fitness and diet adjustments
            if include_fitness_diet:
                if fitness_level == "Excellent":
                    life_expectancy += 7
                elif fitness_level == "Good":
                    life_expectancy += 4
                elif fitness_level == "Poor":
                    life_expectancy -= 4
                
                if diet_rating == "Excellent":
                    life_expectancy += 6
                elif diet_rating == "Good":
                    life_expectancy += 3
                elif diet_rating == "Poor":
                    life_expectancy -= 5
            
            # Random adjustment for unpredictability
            life_expectancy += random.randint(-3, 3)
            
            # Calculate remaining years
            remaining_years = max(0, life_expectancy - age)
            
            # Calculate death date
            death_date = today + datetime.timedelta(days=remaining_years * 365)
            
            # Save the death date to the database - using email as unique identifier
            theclock_data = {
                "user_id": str(user["_id"]),
                "user_email": user_email.lower().strip(),  # Add email for easy lookup
                "death_date": death_date.strftime("%Y-%m-%d"),
                "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
            mongo.db.theclock.insert_one(theclock_data)
            
            return jsonify({
                "message": "Your death date has been calculated. It is now set in stone.",
                "death_date": theclock_data["death_date"],
                "years_remaining": remaining_years,
                "days_remaining": int(remaining_years * 365)
            }), 200
            
        except Exception as e:
            current_app.logger.error(f"Death Clock error: {str(e)}")
            return jsonify({"error": "Something went wrong calculating your death date."}), 500
    
    # Handle GET request (default response)
    return jsonify({"message": "Use POST to calculate your death date"}), 200

@theclock_bp.route("/<email>", methods=["GET"])
@cross_origin(origins=["*"], allow_headers=["Content-Type", "Authorization"],
              methods=["GET"], supports_credentials=False)
def get_death_date(email):
    try:
        # Check if the user exists
        user = mongo.db.users.find_one({"email": email.lower().strip()})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Check if death date exists - search by email for consistency
        existing_entry = mongo.db.theclock.find_one({"user_email": email.lower().strip()})
        if not existing_entry:
            return jsonify({"error": "Death date not calculated yet"}), 404
        
        # Format date for frontend consistency
        created_at = existing_entry.get("created_at")
        if isinstance(created_at, datetime.datetime):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            "message": "Your death date was already determined.",
            "death_date": existing_entry["death_date"],
            "created_at": created_at
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error fetching death date: {str(e)}")
        return jsonify({"error": "Something went wrong fetching your death date."}), 500