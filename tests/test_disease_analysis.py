#!/usr/bin/env python3
"""
Test script for disease analysis functionality
"""

import asyncio
import json
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.alleai_service import AlleAIService

async def test_disease_analysis():
    """Test the disease analysis functionality"""
    
    print("=== Testing Disease Analysis ===")
    
    # Initialize the service
    service = AlleAIService()
    
    # Test parameters
    disease_name = "cashew_leafminer"
    confidence = 0.997
    crop_type = "cashew"
    
    print(f"Testing analysis for: {disease_name}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Crop Type: {crop_type}")
    print()
    
    try:
        # Get disease recommendations
        print("Getting disease recommendations...")
        recommendations = await service.get_disease_recommendations(disease_name, confidence, crop_type)
        
        print("=== RESULTS ===")
        print(f"Disease Overview: {recommendations.get('disease_overview', 'Not found')}")
        print()
        print(f"Immediate Actions: {recommendations.get('immediate_actions', 'Not found')}")
        print()
        print(f"Treatment Protocols:")
        protocols = recommendations.get('treatment_protocols', {})
        if isinstance(protocols, dict):
            for key, value in protocols.items():
                print(f"  {key}: {value}")
        print()
        print(f"Prevention: {recommendations.get('prevention', 'Not found')}")
        print()
        print(f"Monitoring: {recommendations.get('monitoring', 'Not found')}")
        print()
        print(f"Cost Effective: {recommendations.get('cost_effective', 'Not found')}")
        print()
        print(f"Severity Level: {recommendations.get('severity_level', 'Not found')}")
        print()
        print(f"Professional Help: {recommendations.get('professional_help', 'Not found')}")
        
        # Test if it's specific to cashew leafminer
        overview = recommendations.get('disease_overview', '').lower()
        if 'cashew' in overview and 'leafminer' in overview:
            print("\n✅ SUCCESS: Response is specific to cashew leafminer")
        else:
            print("\n❌ WARNING: Response may not be specific to cashew leafminer")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_disease_analysis())
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 