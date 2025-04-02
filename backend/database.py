from flask_pymongo import PyMongo

mongo = PyMongo()

def init_db(app):
    """Initialize MongoDB connection with error handling"""
    mongo.init_app(app)
    
    try:
        # Test the connection by listing collections
        mongo.db.list_collection_names()
        print("✅ MongoDB Connected Successfully!")
    except Exception as e:
        print("❌ MongoDB Connection Failed!", e)
    
    return mongo.db  # Returns database instance
