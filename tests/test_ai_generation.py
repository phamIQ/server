#!/usr/bin/env python3
"""
Test script for AI generation functionality
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.alleai_service import AlleAIService

async def test_ai_generation():
    """Test the AI generation functionality"""
    
    print("=== Testing AI Generation ===")
    
    # Initialize the service
    service = AlleAIService()
    
    print(f"AlleAI available: {service.is_available()}")
    print(f"API Key (first 10 chars): {service.api_key[:10] if service.api_key else 'None'}...")
    print()
    
    # Test title generation
    print("Testing title generation...")
    try:
        title_prompt = "Generate a concise, professional title for this topic: cashew leafminer disease outbreak. The title should be clear, descriptive, and suitable for agricultural content. Return only the title as plain text, no JSON formatting, no quotes, no additional text."
        
        title_result = await service.chat_with_ai(
            user_message=title_prompt,
            models=["gpt-4o"]
        )
        
        print(f"✅ Title generation result: {title_result}")
        print()
        
    except Exception as e:
        print(f"❌ Title generation failed: {str(e)}")
        print()
    
    # Test description generation
    print("Testing description generation...")
    try:
        description_prompt = "Write a comprehensive, informative description for this agricultural topic: cashew leafminer disease. The description should be 2-3 paragraphs long, include practical information, and be written in a professional but accessible tone suitable for farmers and agricultural professionals. Focus on practical applications and benefits. Return the description as plain text, no JSON formatting, no markdown, just natural text."
        
        description_result = await service.chat_with_ai(
            user_message=description_prompt,
            models=["gpt-4o"]
        )
        
        print(f"✅ Description generation result: {description_result}")
        print()
        
    except Exception as e:
        print(f"❌ Description generation failed: {str(e)}")
        print()
    
    # Test general chat
    print("Testing general chat...")
    try:
        chat_result = await service.chat_with_ai(
            user_message="Tell me about cashew leafminer disease in simple terms",
            models=["gpt-4o"]
        )
        
        print(f"✅ Chat result: {chat_result}")
        print()
        
    except Exception as e:
        print(f"❌ Chat failed: {str(e)}")
        print()

if __name__ == "__main__":
    success = asyncio.run(test_ai_generation())
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 