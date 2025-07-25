#!/usr/bin/env python3
"""
Simple test script for the PhamIQ authentication system with MongoDB
"""

import requests
import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print("❌ Backend health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend. Make sure it's running on {BASE_URL}")
        return False

def test_register():
    """Test user registration"""
    user_data = {
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "password": "testpassword123",
        "confirm_password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
        print(f"Register: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Registration successful")
            print(f"Token: {data['access_token'][:50]}...")
            return data['access_token']
        else:
            print(f"❌ Registration failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return None

def test_login():
    """Test user login"""
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Login: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Login successful")
            print(f"Token: {data['access_token'][:50]}...")
            return data['access_token']
        else:
            print(f"❌ Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_me_endpoint(token):
    """Test the /auth/me endpoint"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        print(f"Me endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ User info retrieved successfully")
            print(f"User: {data['first_name']} {data['last_name']} ({data['email']})")
            return True
        else:
            print(f"❌ Me endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Me endpoint error: {e}")
        return False

def test_verify_endpoint(token):
    """Test the /auth/verify endpoint"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/auth/verify", headers=headers)
        print(f"Verify endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Token verification successful")
            print(f"Valid: {data['valid']}")
            return True
        else:
            print(f"❌ Verify endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Verify endpoint error: {e}")
        return False

async def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
        db = client.phamiq
        await db.command("ping")
        print("✅ MongoDB connection successful")
        client.close()
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"Make sure MongoDB is running on {os.getenv('MONGODB_URL', 'localhost:27017')}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing PhamIQ Authentication System (MongoDB)")
    print("=" * 60)
    
    # Test 0: MongoDB connection
    print("Testing MongoDB connection...")
    if not asyncio.run(test_mongodb_connection()):
        print("❌ MongoDB is not available. Please start MongoDB first.")
        return
    
    print("\n" + "=" * 60)
    
    # Test 1: Health check
    if not test_health():
        return
    
    print("\n" + "=" * 60)
    
    # Test 2: Registration
    token = test_register()
    if not token:
        print("\nTrying login instead...")
        token = test_login()
    
    if not token:
        print("❌ No valid token obtained")
        return
    
    print("\n" + "=" * 60)
    
    # Test 3: User info
    test_me_endpoint(token)
    
    print("\n" + "=" * 60)
    
    # Test 4: Token verification
    test_verify_endpoint(token)
    
    print("\n" + "=" * 60)
    print("✅ Authentication system test completed!")

if __name__ == "__main__":
    main() 