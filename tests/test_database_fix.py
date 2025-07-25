#!/usr/bin/env python3
"""
Test script to verify database connectivity and history saving
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.models.database import connect_to_mongo, close_mongo_connection, PredictionHistoryModel
    from app.config import settings
    
    print("✓ Database modules imported successfully")
    
    async def test_database():
        print(f"✓ Testing MongoDB connection to: {settings.MONGODB_URL}")
        print(f"✓ Database name: {settings.MONGODB_DB_NAME}")
        
        # Test connection
        await connect_to_mongo()
        
        # Test creating a sample history entry
        try:
            from bson import ObjectId
            
            # Create a test user ID
            test_user_id = str(ObjectId())
            print(f"✓ Test user ID: {test_user_id}")
            
            # Create sample history entry
            history_entry = await PredictionHistoryModel.create(
                user_id=test_user_id,
                filename="test_image.jpg",
                disease="Tomato Leaf Blight",
                confidence=0.85,
                severity="Moderate",
                crop_type="Tomato"
            )
            
            print("✓ Successfully created history entry")
            print(f"✓ Entry ID: {history_entry.id}")
            print(f"✓ Entry data: {history_entry.to_dict()}")
            
            # Test retrieving history
            history_list = await PredictionHistoryModel.find_by_user_id(test_user_id, limit=10)
            print(f"✓ Retrieved {len(history_list)} history entries")
            
            return True
            
        except Exception as e:
            print(f"✗ Database test failed: {str(e)}")
            return False
        finally:
            await close_mongo_connection()
    
    # Run the test
    success = asyncio.run(test_database())
    
    if success:
        print("✓ All database tests passed - history saving should work")
    else:
        print("✗ Database tests failed - history saving may not work")
        print("  - Check if MongoDB is running")
        print("  - Check MongoDB connection string")
        print("  - Check if MongoDB dependencies are installed")
        
except ImportError as e:
    print(f"✗ Import error: {str(e)}")
    print("  - Make sure all dependencies are installed: pip install -r requirements.txt")
except Exception as e:
    print(f"✗ Unexpected error: {str(e)}") 