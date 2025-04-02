import os
import joblib
import pandas as pd
import numpy as np
import re
import random
from datetime import datetime
from collections import Counter

# Load models and metadata
MODELS_DIR = os.path.join("ml_models")
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))

# Define Symptoms and Dataset
symptoms = metadata["symptoms"]
prediction_classes = label_encoder.classes_
data_path = os.path.join("datasets", "Training.csv")
dataset = pd.read_csv(data_path).dropna(axis=1)
dataset.columns = dataset.columns.str.lower()  # Normalize columns

# Define symptom groups with friendly names and questions
symptom_groups = {
    "general": {
        "symptoms": ["fever", "fatigue", "weight_loss", "restlessness", "lethargy", "sweating", "chills"],
        "questions": [
            "How are you feeling overall? Any fever, fatigue, or general discomfort?",
            "Have you been experiencing any unusual tiredness or changes in energy lately?",
            "Tell me about any general symptoms you've noticed recently."
        ]
    },
    "pain": {
        "symptoms": ["headache", "body_pain", "joint_pain", "chest_pain", "abdominal_pain", "back_pain", "muscle_weakness"],
        "questions": [
            "Are you experiencing any pain? If so, where?",
            "Can you describe any aches, pains, or discomfort you're feeling?",
            "Have you noticed any soreness or pain in specific areas of your body?"
        ]
    },
    "skin": {
        "symptoms": ["itching", "skin_rash", "yellowish_skin", "patches_in_throat", "pus_filled_pimples", "blackheads"],
        "questions": [
            "Have you noticed any changes in your skin, like rashes or unusual coloration?",
            "Are you experiencing any skin issues such as itching or unusual spots?",
            "Tell me about any skin symptoms you've observed."
        ]
    },
    "digestive": {
        "symptoms": ["nausea", "vomiting", "diarrhoea", "constipation", "stomach_pain", "loss_of_appetite", "indigestion"],
        "questions": [
            "How's your appetite and digestion been lately?",
            "Have you experienced any stomach issues, nausea, or changes in bowel movements?",
            "Tell me about any digestive problems you might be having."
        ]
    },
    "respiratory": {
        "symptoms": ["cough", "breathlessness", "phlegm", "throat_irritation", "fast_heart_rate", "chest_pain"],
        "questions": [
            "How's your breathing? Any cough or respiratory issues?",
            "Have you been experiencing any difficulty breathing or throat discomfort?",
            "Tell me about any symptoms related to your breathing or chest."
        ]
    },
    "eyes_ears": {
        "symptoms": ["blurred_vision", "redness_of_eyes", "sinus_pressure", "runny_nose", "earache", "yellowing_of_eyes"],
        "questions": [
            "Have you noticed any changes in your vision or issues with your eyes or ears?",
            "Are you experiencing any sinus problems, runny nose, or ear discomfort?",
            "Tell me about any symptoms related to your eyes, ears, or sinuses."
        ]
    }
}

# Symptom keywords for natural language processing
symptom_keywords = {
    "fever": ["fever", "high temperature", "hot", "burning up"],
    "fatigue": ["tired", "fatigue", "exhausted", "no energy", "weakness"],
    "headache": ["headache", "head pain", "migraine", "head hurts"],
    "skin_rash": ["rash", "skin eruption", "hives", "spots"],
    "cough": ["cough", "coughing", "hacking"],
    "nausea": ["nausea", "feel sick", "queasy", "want to vomit"],
    "vomiting": ["vomit", "throwing up", "puking"],
    "diarrhoea": ["diarrhea", "diarrhoea", "loose stools", "loose bowels"],
    "chest_pain": ["chest pain", "chest tightness", "chest discomfort"],
    "breathlessness": ["shortness of breath", "can't breathe", "hard to breathe", "breathless"],
    "joint_pain": ["joint pain", "painful joints", "sore joints", "arthritis"],
    "abdominal_pain": ["stomach pain", "abdominal pain", "tummy hurts", "belly pain"],
    "back_pain": ["back pain", "backache", "sore back", "back hurts"],
    # Add more mappings as needed
}

# Follow-up questions based on reported symptoms
followup_questions = {
    "fever": ["How high is your fever?", "When did the fever start?", "Does it come and go or is it constant?"],
    "headache": ["Where exactly is the pain?", "How would you rate the pain on a scale of 1-10?", "Does anything make it better or worse?"],
    "cough": ["Is it a dry cough or are you coughing up phlegm?", "How long have you been coughing?", "Is it worse at night or during the day?"],
    "nausea": ["Are you also vomiting?", "Is it related to eating?", "How long have you felt nauseated?"],
    "joint_pain": ["Which joints are affected?", "Is there any swelling?", "Does movement make it worse?"],
    # Add more as needed
}

# Response variations to make the chat more human-like
responses = {
    "acknowledgment": [
        "I understand.",
        "Got it.",
        "I see.",
        "Thanks for sharing that.",
        "Thank you for letting me know."
    ],
    "next_question": [
        "Let me ask you something else.",
        "I'd like to know more about your symptoms.",
        "Let's continue with a few more questions.",
        "I need to ask you about a different area now.",
        "Let's explore another aspect of how you're feeling."
    ],
    "clarification": [
        "Could you tell me more about that?",
        "I'd like to understand that better.",
        "Can you elaborate on that symptom?",
        "How exactly would you describe that feeling?",
        "When did you first notice that?"
    ],
    "empathy": [
        "That must be difficult for you.",
        "I'm sorry you're experiencing that.",
        "That sounds challenging.",
        "I understand that can be uncomfortable.",
        "That's certainly concerning."
    ]
}

# Store user sessions
user_sessions = {}

def start_chat(session_id):
    """Start a new chat session with a friendly introduction"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_sessions[session_id] = {
        "start_time": timestamp,
        "current_group": None,
        "asked_groups": [],
        "reported_symptoms": [],
        "conversation_history": [],
        "followup_symptoms": [],
        "possible_diseases": list(prediction_classes),
        "last_message_type": "greeting"
    }
    
    greeting = "Hi there! I'm your virtual health assistant. I'll ask you some questions to help understand what might be going on with your health. Please describe your main symptoms or concerns in your own words."
    user_sessions[session_id]["conversation_history"].append({"role": "assistant", "message": greeting})
    
    return {"message": greeting}

def get_random_response(response_type):
    """Return a random response of the specified type to make conversation more natural"""
    return random.choice(responses[response_type])

def extract_symptoms(text):
    """Extract symptoms from user's natural language input"""
    found_symptoms = []
    text = text.lower()
    
    # Look for direct symptom mentions or their synonyms
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text:
                found_symptoms.append(symptom)
                break
    
    return list(set(found_symptoms))  # Remove duplicates

def process_message(session_id, user_message):
    """Process a message from the user and generate an appropriate response"""
    if session_id not in user_sessions:
        return start_chat(session_id)
    
    session = user_sessions[session_id]
    session["conversation_history"].append({"role": "user", "message": user_message})
    
    # Extract symptoms from user message
    new_symptoms = extract_symptoms(user_message)
    if new_symptoms:
        session["reported_symptoms"].extend(new_symptoms)
        session["reported_symptoms"] = list(set(session["reported_symptoms"]))  # Remove duplicates
        
        # Add symptoms to follow up on
        for symptom in new_symptoms:
            if symptom in followup_questions and symptom not in session["followup_symptoms"]:
                session["followup_symptoms"].append(symptom)
    
    # Determine next action
    response = ""
    
    # If we have followup questions about specific symptoms, ask those first
    if session["followup_symptoms"]:
        symptom = session["followup_symptoms"].pop(0)
        question = random.choice(followup_questions[symptom])
        response = f"{get_random_response('acknowledgment')} {question}"
        session["last_message_type"] = "followup"
    
    # If we haven't gone through all symptom groups, ask about a new group
    elif len(session["asked_groups"]) < len(symptom_groups):
        # Choose a group we haven't asked about yet
        available_groups = [g for g in symptom_groups if g not in session["asked_groups"]]
        if available_groups:
            next_group = available_groups[0]
            session["asked_groups"].append(next_group)
            session["current_group"] = next_group
            
            # Choose a random question from this group
            group_question = random.choice(symptom_groups[next_group]["questions"])
            transition = get_random_response('next_question')
            response = f"{get_random_response('acknowledgment')} {transition} {group_question}"
            session["last_message_type"] = "group_question"
    
    # If we've asked about all groups and no more followups, make a prediction
    else:
        prediction_result = get_disease_prediction(session_id)
        disease = prediction_result["predicted_disease"]
        confidence = prediction_result["confidence"]
        
        # Format the response with some empathy
        response = f"{get_random_response('acknowledgment')} Based on what you've told me, your symptoms are consistent with {disease} (confidence: {confidence:.1f}%). Remember that this is not a definitive diagnosis - please consult with a healthcare professional for proper evaluation and treatment."
        session["last_message_type"] = "prediction"
    
    # Add assistant's response to conversation history
    session["conversation_history"].append({"role": "assistant", "message": response})
    
    return {"message": response}

def get_disease_prediction(session_id):
    """Use ML models to predict the most likely disease"""
    session_data = user_sessions[session_id]
    reported_symptoms = session_data["reported_symptoms"]
    
    # Count symptoms frequency in the conversation
    symptom_counter = Counter(reported_symptoms)
    
    # Convert symptoms to ML input format
    input_vector = np.zeros(len(symptoms))
    for symp in reported_symptoms:
        if symp in symptoms:
            idx = symptoms.index(symp)
            # Set to 1 if the symptom was mentioned at least once
            input_vector[idx] = 1
    
    # Get predictions from all models
    svm_pred = svm_model.predict_proba([input_vector])[0]
    nb_pred = nb_model.predict_proba([input_vector])[0]
    rf_pred = rf_model.predict_proba([input_vector])[0]
    
    # Average probabilities across models
    final_probs = (svm_pred + nb_pred + rf_pred) / 3
    top_indices = np.argsort(final_probs)[-3:][::-1]  # Get top 3 predictions
    
    predicted_disease = prediction_classes[top_indices[0]]
    confidence = final_probs[top_indices[0]] * 100
    
    # Also store alternative diagnoses
    alternatives = [
        {"disease": prediction_classes[idx], "confidence": final_probs[idx] * 100}
        for idx in top_indices[1:]
    ]
    
    session_data["prediction"] = {
        "main": {"disease": predicted_disease, "confidence": confidence},
        "alternatives": alternatives
    }
    
    return {
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "alternatives": alternatives
    }

def get_conversation_summary(session_id):
    """Return a summary of the conversation and diagnosis"""
    if session_id not in user_sessions:
        return {"error": "Session not found"}
    
    session = user_sessions[session_id]
    
    if "prediction" not in session:
        get_disease_prediction(session_id)
    
    return {
        "symptoms_reported": session["reported_symptoms"],
        "diagnosis": session["prediction"]["main"],
        "alternatives": session["prediction"]["alternatives"],
        "conversation": session["conversation_history"]
    }

# Example usage
# session = start_chat("user123")
# print(session["message"])
# 
# response = process_message("user123", "I have a headache and feel feverish")
# print(response["message"])
# 
# response = process_message("user123", "My throat hurts too and I'm really tired")
# print(response["message"])