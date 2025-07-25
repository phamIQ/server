#!/usr/bin/env python3
"""
Simple authentication test to debug the issue
"""

import requests
import json
import os

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def test_complete_auth_flow():
    """Test the complete authentication flow"""
    print("🧪 Testing Complete Authentication Flow")
    print("=" * 50)
    
    # Step 1: Register a user
    print("1. Registering user...")
    register_data = {
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User", 
        "password": "testpassword123",
        "confirm_password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
        print(f"Register status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            token = data['access_token']
            print(f"✅ Registration successful")
            print(f"Token: {token[:50]}...")
        else:
            print(f"❌ Registration failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Step 2: Test /auth/me endpoint
    print("2. Testing /auth/me endpoint...")
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        print(f"Me endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ /auth/me successful")
            print(f"User: {data['first_name']} {data['last_name']} ({data['email']})")
        else:
            print(f"❌ /auth/me failed: {response.text}")
    except Exception as e:
        print(f"❌ /auth/me error: {e}")
    
    print("\n" + "=" * 50)
    
    # Step 3: Test /auth/verify endpoint
    print("3. Testing /auth/verify endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/auth/verify", headers=headers)
        print(f"Verify endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ /auth/verify successful")
            print(f"Valid: {data['valid']}")
        else:
            print(f"❌ /auth/verify failed: {response.text}")
    except Exception as e:
        print(f"❌ /auth/verify error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Authentication flow test completed!")

if __name__ == "__main__":
    test_complete_auth_flow() 