import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")  # Change in production
    DEBUG = False

    # MongoDB Connection
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/meds_ai")  # Default local DB

    # ML Model Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "backend/ml/disease_prediction.pkl")
    VECTOR_PATH = os.getenv("VECTOR_PATH", "backend/ml/vectorizer.pkl")  # If using NLP

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    MONGO_URI = os.getenv("MONGO_URI", "your_mongodb_atlas_uri")  # Use Atlas in production

# Load configuration based on ENV
config_by_name = {
    "dev": DevelopmentConfig,
    "prod": ProductionConfig
}
