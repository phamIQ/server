#!/usr/bin/env python3
"""
Test script to verify AlleAI chat integration
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def test_chat_integration():
    """Test the chat integration with AlleAI"""
    
    print("🧪 Testing AlleAI Chat Integration")
    print("=" * 50)
    
    try:
        from app.services.alleai_service import alleai_service
        
        # Test 1: Service availability
        print("\n1. Testing service availability...")
        is_available = alleai_service.is_available()
        print(f"   Service available: {is_available}")
        
        if not is_available:
            print("   ❌ Service not available - skipping further tests")
            return False
        
        # Test 2: Available models
        print("\n2. Testing available models...")
        models = alleai_service.get_available_models()
        print(f"   Available models: {models}")
        
        # Test 3: Basic chat functionality
        print("\n3. Testing basic chat functionality...")
        try:
            response = await alleai_service.chat_with_ai(
                user_message="Hello, can you help me with plant disease management?",
                models=["gpt-4o"]
            )
            print(f"   ✅ Chat response received (length: {len(response)} characters)")
            print(f"   Preview: {response[:200]}...")
        except Exception as e:
            print(f"   ❌ Chat test failed: {str(e)}")
            return False
        
        # Test 4: Disease recommendations
        print("\n4. Testing disease recommendations...")
        try:
            recommendations = await alleai_service.get_disease_recommendations(
                "Tomato Leaf Blight", 
                0.85, 
                "Tomato",
                models=["gpt-4o"]
            )
            print(f"   ✅ Disease recommendations received")
            print(f"   Keys: {list(recommendations.keys())}")
        except Exception as e:
            print(f"   ❌ Disease recommendations test failed: {str(e)}")
            return False
        
        print("\n✅ All tests passed - AlleAI chat integration is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_chat_integration())
    if success:
        print("\n🎉 AlleAI chat integration is working correctly!")
    else:
        print("\n💥 AlleAI chat integration has issues!") 