import requests

def test_proxy_endpoint():
    """Test the proxy endpoint with a sample image URL"""
    
    # Test with a sample image URL (replace with actual URL from your logs)
    test_image_url = "https://alle-ai-file-server.s3.us-east-1.amazonaws.com/api-generated-content/alle-ai-dall-e-3-1752312887.png"
    
    # Create the proxy URL
    proxy_url = f"http://localhost:8000/ai/proxy-image?url={test_image_url}"
    
    print(f"Testing proxy endpoint with URL: {proxy_url}")
    
    try:
        response = requests.get(proxy_url, stream=True)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Proxy endpoint is working correctly!")
            print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        else:
            print(f"❌ Proxy endpoint failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing proxy endpoint: {str(e)}")

if __name__ == "__main__":
    test_proxy_endpoint() 