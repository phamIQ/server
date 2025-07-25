import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_image_generation():
    """Test AlleAI image generation"""
    
    # Get API key
    api_key = "alle-m5iGU5J6cKxwMp6J0qAHmVXHmbCPNc6hhYY6"
    
    try:
        from alleai.core import AlleAIClient
        
        print(f"Testing AlleAI image generation...")
        print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
        
        # Initialize client
        client = AlleAIClient(api_key=api_key)
        
        # Test image generation
        prompt = "A healthy maize field at sunrise"
        
        print(f"Generating image with prompt: {prompt}")
        
        image_response = client.image.generate({
            "models": ["dall-e-3"],
            "prompt": prompt,
            "n": 1,
            "height": 1024,
            "width": 1024,
            "seed": 42,
            "model_specific_params": {}
        })
        
        print(f"Response type: {type(image_response)}")
        print(f"Response: {image_response}")
        
        # Try to extract URL
        if isinstance(image_response, dict):
            # Check for the actual AlleAI response structure
            if 'responses' in image_response:
                responses = image_response['responses']
                if isinstance(responses, dict) and 'responses' in responses:
                    model_responses = responses['responses']
                    if isinstance(model_responses, dict):
                        # Get the first model's response (usually dall-e-3)
                        for model_name, url in model_responses.items():
                            if isinstance(url, str) and url.startswith('http'):
                                print(f"✅ Image URL found from {model_name}: {url}")
                                # Return proxy URL to avoid CORS issues
                                proxy_url = f"/ai/proxy-image?url={url}"
                                print(f"✅ Proxy URL: {proxy_url}")
                                return proxy_url
            
            # Check for direct responses structure (if the response is already the inner responses object)
            if 'responses' in image_response and isinstance(image_response['responses'], dict):
                model_responses = image_response['responses']
                for model_name, url in model_responses.items():
                    if isinstance(url, str) and url.startswith('http'):
                        print(f"✅ Image URL found from {model_name}: {url}")
                        # Return proxy URL to avoid CORS issues
                        proxy_url = f"/ai/proxy-image?url={url}"
                        print(f"✅ Proxy URL: {proxy_url}")
                        return proxy_url
            
            # Fallback: check for data structure
            if 'data' in image_response:
                data = image_response['data']
                if isinstance(data, list) and len(data) > 0:
                    image_data = data[0]
                    if isinstance(image_data, dict) and 'url' in image_data:
                        print(f"✅ Image URL found: {image_data['url']}")
                        return image_data['url']
            
            # Fallback: try to extract URL from response object
            if 'url' in image_response:
                print(f"✅ Direct URL found: {image_response['url']}")
                return image_response['url']
        
        print("❌ No image URL found in response")
        return None
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = asyncio.run(test_image_generation())
    if result:
        print(f"✅ Image generation successful: {result}")
    else:
        print("❌ Image generation failed") 