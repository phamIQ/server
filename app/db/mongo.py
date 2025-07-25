from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
try:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client["smart_agri"]
    prediction_collection = db["predictions"]
except Exception as e:
    prediction_collection = None
    mongo_error = str(e)
