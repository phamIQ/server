from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import traceback
import re
import json
from fastapi.responses import StreamingResponse, Response
import requests
from app.services.alleai_service import freeai_service
from app.config import settings

router = APIRouter(prefix="/ai", tags=["ai"])

class PromptRequest(BaseModel):
    prompt: str
    models: Optional[List[str]] = None

class ChatResponse(BaseModel):
    result: str

class ImageResponse(BaseModel):
    url: str

def extract_main_text(text: str) -> str:
    # Remove lines that look like metadata (e.g., 'Created at:', 'Date:', etc.)
    lines = text.splitlines()
    filtered = [line for line in lines if not (line.lower().startswith('created at') or line.lower().startswith('date:') or line.lower().startswith('created:'))]
    # Remove lines that look like JSON metadata or cost info
    filtered = [line for line in filtered if not re.search(r'(tokens_used|usage|cost|finish_reason|prompt_tokens|completion_tokens|total_tokens|total_cost|input_cost|output_cost)', line)]
    return '\n'.join(filtered).strip()

def summarize_text(text: str, max_sentences: int = 3) -> str:
    # Simple summarization: return the first max_sentences sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:max_sentences]).strip()

def format_as_blog_post(topic: str, content: str) -> str:
    # Format the content as a professional blog post
    heading = f"# {topic.strip().capitalize()}"
    intro = f"As a professional blog writer, here is a concise and informative post about {topic.strip().lower()}:"
    body = content.strip()
    return f"{heading}\n\n{intro}\n\n{body}"

def is_valid_image_url(url: str) -> bool:
    # Basic check for image URL (http/https and ends with image extension or is a data URL)
    return (
        url.startswith('http://') or url.startswith('https://') or url.startswith('data:image/')
    )

def extract_image_url(response) -> str:
    print("Type of response:", type(response))
    print("Response repr:", repr(response))
    print("Response dir:", dir(response))
    # Try dict-style access
    outer = None
    if isinstance(response, dict):
        outer = response.get("responses")
    else:
        # Try attribute access
        outer = getattr(response, "responses", None)
    if outer:
        inner = None
        if isinstance(outer, dict):
            inner = outer.get("responses")
        else:
            inner = getattr(outer, "responses", None)
        if inner:
            if isinstance(inner, dict):
                for url in inner.values():
                    if isinstance(url, str) and url.startswith("http"):
                        print("Extracted image URL:", url)
                        return url
            else:
                # If it's not a dict, try attribute access
                for attr in dir(inner):
                    value = getattr(inner, attr)
                    if isinstance(value, str) and value.startswith("http"):
                        print("Extracted image URL:", value)
                        return value
    print("Extracted image URL: (placeholder)")
    return settings.NO_IMAGE_AVAILABLE_URL

def clean_text_response(text: str) -> str:
    """Clean text response to ensure it's natural text without JSON formatting"""
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove quotes if the entire response is quoted
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # If it looks like JSON, try to extract the actual content
    if text.startswith('{') and text.endswith('}'):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Look for common text fields
                for key in ['title', 'text', 'content', 'message', 'response', 'result', 'description']:
                    if key in data and isinstance(data[key], str):
                        return data[key].strip()
                
                # If no text field found, try to format the content nicely
                formatted_parts = []
                for key, value in data.items():
                    if isinstance(value, str) and value.strip():
                        # Format the key as a heading
                        formatted_key = key.replace('_', ' ').title()
                        formatted_parts.append(f"**{formatted_key}:**\n{value}")
                
                if formatted_parts:
                    return '\n\n'.join(formatted_parts)
                
                # If no string values found, return the first non-string value
                for value in data.values():
                    if value:
                        return str(value)
        except json.JSONDecodeError:
            pass
    
    # Clean up common AI response artifacts
    text = text.replace('```json', '').replace('```', '').strip()
    text = text.replace('**', '**')  # Ensure markdown formatting is preserved
    
    # Remove any remaining JSON artifacts
    import re
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove JSON objects
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove JSON arrays
    text = re.sub(r'"[^"]*":\s*"[^"]*"', '', text)  # Remove JSON key-value pairs
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive line breaks
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Remove leading/trailing whitespace
    
    return text

@router.post("/generate-title", response_model=ChatResponse)
async def generate_title(request: PromptRequest):
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Please configure AlleAI API key."
            )
        
        # Create a prompt for title generation
        title_prompt = f"Generate a concise, professional title for this topic: {request.prompt}. The title should be clear, descriptive, and suitable for agricultural content. Return ONLY the title as plain text, no JSON formatting, no quotes, no additional text, no markdown."
        
        models = request.models or ["gpt-4o"]
        result = await freeai_service.chat_with_ai(
            user_message=title_prompt,
            conversation_history=[],
            models=models
        )
        
        # Clean the response to get just the title
        title = clean_text_response(result)
        
        return {"result": title}
    except Exception as e:
        print("Error in /ai/generate-title:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-description", response_model=ChatResponse)
async def generate_description(request: PromptRequest):
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Please configure AlleAI API key."
            )
        
        # Create a prompt for description generation
        description_prompt = f"Write a comprehensive, informative description for this agricultural topic: {request.prompt}. The description should be 2-3 paragraphs long, include practical information, and be written in a professional but accessible tone suitable for farmers and agricultural professionals. Focus on practical applications and benefits. Return the description as natural text with proper formatting, no JSON, no markdown, just well-structured paragraphs."
        
        models = request.models or ["gpt-4o"]
        result = await freeai_service.chat_with_ai(
            user_message=description_prompt,
            conversation_history=[],
            models=models
        )
        
        # Clean the response to ensure it's plain text
        description = clean_text_response(result)
        
        return {"result": description}
    except Exception as e:
        print("Error in /ai/generate-description:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: PromptRequest):
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Please configure AlleAI API key."
            )
        
        # Create a prompt for image generation
        image_prompt = f"Generate a high-quality, realistic image of: {request.prompt}. The image should be suitable for agricultural content, professional, and visually appealing. Focus on clarity and realism."
        
        models = request.models or ["dall-e-3"]
        
        # Use the new image generation method
        image_url = await freeai_service.generate_image(image_prompt, models)
        if not image_url or not isinstance(image_url, str):
            image_url = settings.PLACEHOLDER_IMAGE_URL
        
        return {"url": image_url}
    except Exception as e:
        print("Error in /ai/generate-image:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/proxy-image")
def proxy_image(url: str):
    try:
        r = requests.get(url, stream=True)
        # Guess content type from headers, fallback to image/png
        content_type = r.headers.get('content-type', 'image/png')
        headers = {"Access-Control-Allow-Origin": "*"}
        return StreamingResponse(r.raw, media_type=content_type, headers=headers)
    except Exception as e:
        print("Error proxying image:", e)
        return Response(content="Failed to fetch image", status_code=500)

@router.post("/chat")
async def ai_chat(request: PromptRequest):
    """General AI chat endpoint"""
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Please configure AlleAI API key."
            )
        
        # Enhance the prompt to request natural text responses
        enhanced_prompt = f"{request.prompt}\n\nPlease provide a natural, conversational response. Use proper formatting with paragraphs, bullet points where appropriate, and clear explanations. Avoid JSON formatting or technical jargon unless specifically requested."
        
        models = request.models or ["gpt-4o"]
        result = await freeai_service.chat_with_ai(
            user_message=enhanced_prompt,
            conversation_history=[],
            models=models
        )
        
        # Clean the response to ensure it's natural text
        cleaned_result = clean_text_response(result)
        
        return {"result": cleaned_result}
    except Exception as e:
        print("Error in /ai/chat:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disease-analysis")
async def disease_analysis(request: PromptRequest):
    """Specialized endpoint for disease analysis"""
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="AI service is not available. Please configure AlleAI API key."
            )
        
        # Parse the prompt to extract disease information
        # Expected format: "disease_name|confidence|crop_type"
        parts = request.prompt.split('|')
        if len(parts) >= 3:
            disease_name = parts[0].strip()
            confidence = float(parts[1].strip())
            crop_type = parts[2].strip()
            
            models = request.models or ["gpt-4o"]
            result = await freeai_service.get_disease_recommendations(
                disease_name=disease_name,
                confidence=confidence,
                crop_type=crop_type,
                models=models
            )
            
            return {"result": result}
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Expected: disease_name|confidence|crop_type"
            )
    except Exception as e:
        print("Error in /ai/disease-analysis:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def ai_status():
    """Get AI service status"""
    try:
        is_available = freeai_service.is_available()
        models = freeai_service.get_available_models()
        
        return {
            "status": "available" if is_available else "unavailable",
            "models": models,
            "service": "AlleAI"
        }
    except Exception as e:
        print("Error in /ai/status:", e)
        return {
            "status": "error",
            "error": str(e),
            "service": "AlleAI"
        } 

@router.get("/test")
async def test_ai():
    """Simple test endpoint to verify AI service is working"""
    try:
        if not freeai_service.is_available():
            return {
                "status": "error",
                "message": "AlleAI service not available",
                "api_key_configured": bool(freeai_service.api_key)
            }
        
        # Test with a simple prompt
        result = await freeai_service.chat_with_ai(
            user_message="Say 'Hello, AI is working!'",
            models=["gpt-4o"]
        )
        
        return {
            "status": "success",
            "message": "AI service is working",
            "test_response": result,
            "api_key_configured": bool(freeai_service.api_key)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"AI service test failed: {str(e)}",
            "api_key_configured": bool(freeai_service.api_key)
        } 