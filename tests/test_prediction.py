#!/usr/bin/env python3
"""
Test script for the prediction endpoint
"""

import requests
import json
import os

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def test_prediction_flow():
    """Test the complete prediction flow"""
    print("🧪 Testing Prediction Flow")
    print("=" * 50)
    
    # Step 1: Login to get token
    print("1. Logging in...")
    login_data = {
        "email": "test@example.com",
        "password": "testpassword123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Login status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            token = data['access_token']
            print(f"✅ Login successful")
            print(f"Token: {token[:50]}...")
        else:
            print(f"❌ Login failed: {response.text}")
            print("Trying registration...")
            
            # Try registration instead
            register_data = {
                "email": "test@example.com",
                "first_name": "Test",
                "last_name": "User",
                "password": "testpassword123",
                "confirm_password": "testpassword123"
            }
            
            response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
            if response.status_code == 200:
                data = response.json()
                token = data['access_token']
                print(f"✅ Registration successful")
                print(f"Token: {token[:50]}...")
            else:
                print(f"❌ Registration failed: {response.text}")
                return
    except Exception as e:
        print(f"❌ Login/Registration error: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Step 2: Test prediction endpoint
    print("2. Testing prediction endpoint...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a simple test image (you can replace this with a real image file)
    test_image_path = "test_image.jpg"
    
    # Check if test image exists, if not create a simple one
    if not os.path.exists(test_image_path):
        print("Creating test image...")
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='green')
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill='darkgreen')
        img.save(test_image_path)
        print(f"✅ Created test image: {test_image_path}")
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict/?top_k=3", headers=headers, files=files)
        
        print(f"Prediction status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"Status: {data['status']}")
            print(f"Total classes: {data['total_classes']}")
            print("Predictions:")
            for i, pred in enumerate(data['predictions']):
                print(f"  {i+1}. {pred['class_name']} - {pred['confidence_percentage']}")
        else:
            print(f"❌ Prediction failed: {response.text}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Prediction flow test completed!")

if __name__ == "__main__":
    test_prediction_flow() 