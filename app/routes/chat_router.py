from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import os
from app.config import settings

router = APIRouter(prefix="/chat", tags=["Chat"])

SITE_URL = "https://phamiq.ai"  # Update as needed
SITE_NAME = "Phamiq AI"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY or "",
)

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

@router.post("/", response_model=ChatResponse)
async def chat_with_openrouter(request: ChatRequest):
    try:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            model="openai/gpt-4o",
            messages=messages,
            max_tokens=500  # Lowered to avoid 402 errors
        )
        return ChatResponse(response=completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter error: {str(e)}") 