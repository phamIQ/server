from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from bson import ObjectId
from datetime import datetime
from typing import Optional, List
from app.config import settings
from pydantic import BaseModel, Field
from app.utils.passwords import get_password_hash, verify_password

# MongoDB client
client: Optional[AsyncIOMotorClient] = None
database = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client, database
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        if client is None:
            raise Exception("Failed to create MongoDB client")
        database = client[settings.MONGODB_DB_NAME]
        
        # Test the connection
        await client.admin.command('ping')
        
        # Create indexes for better performance
        await database.users.create_index([("email", ASCENDING)], unique=True)
        await database.prediction_history.create_index([("user_id", ASCENDING)])
        await database.prediction_history.create_index([("created_at", ASCENDING)])
        await database.chat_history.create_index([("user_id", ASCENDING)])
        await database.chat_history.create_index([("created_at", ASCENDING)])
        print("Connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        print("Running without database - some features may not work")
        client = None
        database = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("Disconnected from MongoDB")

def get_database():
    """Get database instance"""
    return database

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema

class PredictionHistory(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    filename: str
    disease: str
    confidence: float
    severity: str
    crop_type: str
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}

class ChatHistory(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    title: str
    messages: List[ChatMessage]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class AnalysisJob(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[dict] = None
    error: Optional[str] = None
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class AnalysisJobModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('_id')
        self.user_id = kwargs.get('user_id')
        self.status = kwargs.get('status', 'pending')
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.result = kwargs.get('result')
        self.error = kwargs.get('error')

    @classmethod
    async def create(cls, user_id: str):
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        job_data = {
            "user_id": ObjectId(user_id),
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "result": None,
            "error": None
        }
        result = await db.analysis_jobs.insert_one(job_data)
        job_data['_id'] = result.inserted_id
        return cls(**job_data)

    @classmethod
    async def find_by_id(cls, job_id: str):
        db = get_database()
        if db is None:
            return None
        try:
            object_id = ObjectId(job_id)
            job_data = await db.analysis_jobs.find_one({"_id": object_id})
            if job_data:
                return cls(**job_data)
        except Exception as e:
            print(f"Error finding analysis job by ID: {e}")
        return None

    async def update_status(self, status: str, result: Optional[dict] = None, error: Optional[str] = None):
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        self.status = status
        self.updated_at = datetime.utcnow()
        update_data = {"status": status, "updated_at": self.updated_at}
        if result is not None:
            self.result = result
            update_data["result"] = result
        if error is not None:
            self.error = error
            update_data["error"] = error
        await db.analysis_jobs.update_one({"_id": self.id}, {"$set": update_data})

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error
        }

# User model for MongoDB
class User:
    def __init__(self, **kwargs):
        self.id = kwargs.get('_id')
        self.email = kwargs.get('email')
        self.first_name = kwargs.get('first_name')
        self.last_name = kwargs.get('last_name')
        self.hashed_password = kwargs.get('hashed_password')
        self.is_active = kwargs.get('is_active', True)
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.location = kwargs.get('location')
        self.language = kwargs.get('language')
        self.timezone = kwargs.get('timezone')
        self.notifications = kwargs.get('notifications', {"analysis_results": True, "disease_alerts": True})
    
    @classmethod
    async def create(cls, **kwargs):
        """Create a new user"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        user_data = {
            "email": kwargs['email'],
            "first_name": kwargs['first_name'],
            "last_name": kwargs['last_name'],
            "hashed_password": kwargs['hashed_password'],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.users.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return cls(**user_data)
    
    @classmethod
    async def find_by_email(cls, email: str):
        """Find user by email"""
        db = get_database()
        if db is None:
            return None
        user_data = await db.users.find_one({"email": email})
        if user_data:
            return cls(**user_data)
        return None
    
    @classmethod
    async def find_by_id(cls, user_id: str):
        """Find user by ID"""
        db = get_database()
        if db is None:
            return None
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(user_id)
            user_data = await db.users.find_one({"_id": object_id})
            if user_data:
                return cls(**user_data)
        except Exception as e:
            print(f"Error finding user by ID: {e}")
        return None
    
    async def save(self):
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        
        # Only update the actual user fields, not computed fields
        update_data = {
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "hashed_password": self.hashed_password,
            "is_active": self.is_active,
            "updated_at": datetime.utcnow(),
            "location": self.location,
            "language": self.language,
            "timezone": self.timezone,
            "notifications": self.notifications
        }
        
        await db.users.update_one({"_id": self.id}, {"$set": update_data})

    async def update_profile(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)
        await self.save()

    async def set_password(self, new_password):
        self.hashed_password = get_password_hash(new_password)
        await self.save()

    def verify_password(self, password):
        if not self.hashed_password:
            return False
        return verify_password(password, self.hashed_password)

    async def delete(self):
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        await db.users.delete_one({"_id": self.id})
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "location": self.location,
            "language": self.language,
            "timezone": self.timezone,
            "notifications": self.notifications
        }

# ChatHistory model for MongoDB
class ChatHistoryModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('_id')
        self.user_id = kwargs.get('user_id')
        self.title = kwargs.get('title')
        self.messages = kwargs.get('messages', [])
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
    
    @classmethod
    async def create(cls, user_id: str, title: str, messages: List[dict]):
        """Create a new chat history"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        
        chat_data = {
            "user_id": ObjectId(user_id),
            "title": title,
            "messages": messages,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await db.chat_history.insert_one(chat_data)
        chat_data['_id'] = result.inserted_id
        return cls(**chat_data)
    
    @classmethod
    async def find_by_user_id(cls, user_id: str, limit: int = 50):
        """Find chat history by user ID"""
        db = get_database()
        if db is None:
            return []
        
        try:
            object_id = ObjectId(user_id)
            cursor = db.chat_history.find({"user_id": object_id}).sort("updated_at", -1).limit(limit)
            chats = await cursor.to_list(length=limit)
            return [cls(**chat) for chat in chats]
        except Exception as e:
            print(f"Error finding chat history: {e}")
            return []
    
    @classmethod
    async def find_by_id(cls, chat_id: str):
        """Find chat history by ID"""
        db = get_database()
        if db is None:
            return None
        
        try:
            object_id = ObjectId(chat_id)
            chat_data = await db.chat_history.find_one({"_id": object_id})
            if chat_data:
                return cls(**chat_data)
        except Exception as e:
            print(f"Error finding chat by ID: {e}")
        return None
    
    async def update_messages(self, messages: List[dict]):
        """Update chat messages"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        
        self.messages = messages
        self.updated_at = datetime.utcnow()
        
        await db.chat_history.update_one(
            {"_id": self.id},
            {"$set": {"messages": messages, "updated_at": self.updated_at}}
        )
    
    async def update_title(self, title: str):
        """Update chat title"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        
        self.title = title
        self.updated_at = datetime.utcnow()
        
        await db.chat_history.update_one(
            {"_id": self.id},
            {"$set": {"title": title, "updated_at": self.updated_at}}
        )
    
    async def delete(self):
        """Delete chat history"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        await db.chat_history.delete_one({"_id": self.id})
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# PredictionHistory model for MongoDB
class PredictionHistoryModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get('_id')
        self.user_id = kwargs.get('user_id')
        self.filename = kwargs.get('filename')
        self.disease = kwargs.get('disease')
        self.confidence = kwargs.get('confidence')
        self.severity = kwargs.get('severity')
        self.crop_type = kwargs.get('crop_type')
        self.image_url = kwargs.get('image_url')
        self.recommendations = kwargs.get('recommendations')  # Store LLM recommendations
        self.created_at = kwargs.get('created_at', datetime.utcnow())
    
    @classmethod
    async def create(cls, **kwargs):
        """Create a new prediction history entry"""
        db = get_database()
        
        # Convert user_id string to ObjectId
        try:
            user_id_obj = ObjectId(kwargs['user_id'])
        except Exception as e:
            raise Exception(f"Invalid user_id format: {str(e)}")
        
        history_data = {
            "user_id": user_id_obj,
            "filename": kwargs['filename'],
            "disease": kwargs['disease'],
            "confidence": kwargs['confidence'],
            "severity": kwargs['severity'],
            "crop_type": kwargs['crop_type'],
            "image_url": kwargs.get('image_url'),
            "recommendations": kwargs.get('recommendations'),  # Store LLM recommendations
            "created_at": datetime.utcnow()
        }
        if db is None:
            raise Exception("Database not connected")
        result = await db.prediction_history.insert_one(history_data)
        history_data['_id'] = result.inserted_id
        return cls(**history_data)
    
    @classmethod
    async def find_by_user_id(cls, user_id: str, limit: int = 50):
        """Find prediction history for a user"""
        db = get_database()
        if db is None:
            return []
        try:
            object_id = ObjectId(user_id)
            cursor = db.prediction_history.find(
                {"user_id": object_id}
            ).sort("created_at", -1).limit(limit)
            
            history_list = []
            async for doc in cursor:
                history_list.append(cls(**doc))
            return history_list
        except Exception as e:
            print(f"Error finding prediction history: {e}")
            return []
    
    @classmethod
    async def find_by_id(cls, history_id: str):
        """Find prediction history by ID"""
        db = get_database()
        if db is None:
            return None
        
        try:
            object_id = ObjectId(history_id)
            history_data = await db.prediction_history.find_one({"_id": object_id})
            if history_data:
                return cls(**history_data)
        except Exception as e:
            print(f"Error finding history by ID: {e}")
        return None
    
    async def delete(self):
        """Delete prediction history entry"""
        db = get_database()
        if db is None:
            raise Exception("Database not connected")
        await db.prediction_history.delete_one({"_id": self.id})
    
    def to_dict(self):
        """Convert prediction history to dictionary"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "filename": self.filename,
            "disease": self.disease,
            "confidence": self.confidence,
            "severity": self.severity,
            "crop_type": self.crop_type,
            "image_url": self.image_url,
            "recommendations": self.recommendations,  # Include LLM recommendations
            "created_at": self.created_at
        } 