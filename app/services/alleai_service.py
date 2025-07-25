import os
import logging
import json
from typing import Dict, Any, Optional, List
from app.config import settings
import asyncio
import re

logger = logging.getLogger(__name__)

class FreeAIService:
    """Service for handling free/local AI interactions with agricultural expertise"""

    def __init__(self):
        self.default_models = ["gpt-4o", "yi-large"]
        self._recommendations_cache = {}
        logger.info(f"Initializing FreeAI service (no paid API required)...")

    def is_available(self) -> bool:
        return True

    def clear_cache(self):
        self._recommendations_cache.clear()
        logger.info("Recommendations cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._recommendations_cache),
            "cached_keys": list(self._recommendations_cache.keys())
        }

    def get_available_models(self) -> List[str]:
        return self.default_models

    async def test_connection(self) -> bool:
        logger.info("FreeAI connection test always successful (no paid API required)")
        return True

    def _get_agricultural_system_prompt(self) -> str:
        return """You are an expert agricultural AI assistant specializing in crop disease management and agricultural best practices. You provide helpful, conversational advice for farmers and agricultural professionals.\n\n**Your Expertise Areas:**\n- Plant disease identification and management\n- Soil health and fertilization\n- Irrigation and watering practices\n- Pest control strategies\n- Crop management and rotation\n- Organic farming methods\n- Climate-smart agriculture\n\n**Response Guidelines:**\n1. **Provide natural, conversational responses** like ChatGPT\n2. **Give practical, actionable advice** that farmers can implement\n3. **Include both organic and conventional solutions** when applicable\n4. **Consider local conditions** and climate factors\n5. **Provide step-by-step instructions** for complex procedures\n6. **Include safety warnings** for chemical treatments\n7. **Suggest monitoring and follow-up actions**\n\n**Important:**\n- Respond naturally and conversationally, not in structured JSON format\n- Use clear, helpful language that's easy to understand\n- Provide specific, actionable advice\n- Be friendly and supportive in your tone\n- Focus on practical solutions that farmers can implement immediately\n- Always return plain text responses, never JSON format\n- For title generation, return only the title as plain text\n- For description generation, return natural paragraphs as plain text"""

    def _get_disease_analysis_prompt(self, disease_name: str, confidence: float, crop_type: str) -> str:
        return f"""You are an expert agricultural scientist specializing in crop disease management.\n\n**Analysis Request:**\n- Disease: {disease_name}\n- Confidence Level: {confidence:.1%}\n- Crop Type: {crop_type}\n\nPlease provide a comprehensive analysis for {disease_name} affecting {crop_type} plants. \n\n**IMPORTANT:** Return your response as a valid JSON object with this exact structure:\n\n{{\n    \"disease_overview\": \"Detailed description of {disease_name}, its symptoms, and how it affects {crop_type} plants\",\n    \"immediate_actions\": \"Step-by-step immediate response plan for {disease_name}\",\n    \"treatment_protocols\": {{\n        \"organic\": \"Organic treatment methods for {disease_name}\",\n        \"chemical\": \"Chemical treatment options for {disease_name} if applicable\",\n        \"application\": \"How and when to apply treatments for {disease_name}\"\n    }},\n    \"prevention\": \"Long-term prevention strategies for {disease_name}\",\n    \"monitoring\": \"How to monitor progress and effectiveness of treatments for {disease_name}\",\n    \"cost_effective\": \"Budget-friendly solutions for {disease_name}\",\n    \"severity_level\": \"Low/Moderate/High based on {disease_name}\",\n    \"professional_help\": \"When to consult agricultural experts for {disease_name}\"\n}}\n\n**CRITICAL:** \n- Respond ONLY with the JSON object above\n- Do NOT include any text before or after the JSON\n- Make the response specific to {disease_name} and {crop_type}\n- Provide practical, actionable advice for farmers\n"""

    def _get_fallback_response(self, user_message: str) -> str:
        message_lower = user_message.lower()
        
        # Return natural conversational responses instead of JSON
        if "disease" in message_lower or "sick" in message_lower or "problem" in message_lower:
            return "I understand you're dealing with plant health issues. Here are some general steps you can take:\n\n1. **Isolate affected plants** to prevent the problem from spreading to healthy plants\n2. **Remove and destroy severely infected parts** - this helps stop the spread\n3. **Improve air circulation** around your plants by ensuring proper spacing\n4. **Avoid overhead watering** which can spread diseases\n5. **Use clean tools** when working with different plants\n\nFor more specific advice, I'd need to know what type of plant you're working with and what symptoms you're seeing. Could you tell me more about your specific situation?"
        
        elif "soil" in message_lower or "fertilizer" in message_lower or "nutrient" in message_lower:
            return "Soil health is crucial for plant growth! Here are some key points:\n\n• **Test your soil** to understand its current nutrient levels\n• **Add organic matter** like compost to improve soil structure\n• **Use balanced fertilizers** based on your soil test results\n• **Consider crop rotation** to maintain soil health\n• **Monitor pH levels** - most plants prefer slightly acidic to neutral soil\n\nWhat type of soil are you working with, and what are you trying to grow?"
        
        elif "water" in message_lower or "irrigation" in message_lower:
            return "Proper watering is essential for plant health! Here are some tips:\n\n• **Water deeply but less frequently** to encourage deep root growth\n• **Water early in the morning** to reduce evaporation and disease risk\n• **Check soil moisture** before watering - stick your finger in the soil\n• **Use mulch** to retain soil moisture\n• **Avoid overwatering** which can cause root rot\n\nWhat's your current watering schedule, and are you seeing any specific issues?"
        
        elif "pest" in message_lower or "insect" in message_lower or "bug" in message_lower:
            return "Pest management can be challenging! Here's a balanced approach:\n\n• **Identify the pest first** - this helps determine the best control method\n• **Start with cultural controls** like removing affected plants\n• **Use beneficial insects** when possible\n• **Consider organic options** like neem oil before chemical pesticides\n• **Monitor regularly** to catch problems early\n\nCan you describe what pests you're seeing and on what plants?"
        
        elif "treatment" in message_lower or "cure" in message_lower or "fix" in message_lower:
            return "For plant treatment, here's a systematic approach:\n\n1. **Identify the problem** - disease, pest, or environmental issue\n2. **Choose appropriate treatment** - organic or chemical options\n3. **Apply correctly** - follow label instructions and timing\n4. **Monitor results** - watch for improvement or side effects\n5. **Prevent recurrence** - address underlying causes\n\nWhat specific treatment are you considering, and what problem are you trying to solve?"
        
        elif "prevent" in message_lower or "avoid" in message_lower:
            return "Prevention is always better than cure! Here are key prevention strategies:\n\n• **Choose resistant varieties** when available\n• **Practice crop rotation** to break disease cycles\n• **Maintain good spacing** for air circulation\n• **Use clean tools and equipment**\n• **Monitor regularly** for early detection\n• **Keep garden clean** - remove debris and weeds\n\nWhat specific problem are you trying to prevent?"
        
        elif "organic" in message_lower or "natural" in message_lower:
            return "Organic solutions are great for sustainable farming! Here are some options:\n\n• **Neem oil** - effective against many pests and diseases\n• **Baking soda spray** - helps with fungal issues\n• **Beneficial insects** - ladybugs, lacewings, etc.\n• **Compost tea** - boosts plant immunity\n• **Crop rotation** - naturally breaks pest cycles\n• **Companion planting** - some plants protect others\n\nWhat specific organic solution are you interested in?"
        
        else:
            return "I'm here to help with your agricultural questions! I can assist with plant diseases, soil health, pest management, irrigation, and general farming practices.\n\nWhat specific agricultural challenge are you facing today? I'd be happy to provide some practical advice based on your situation. You can ask me about:\n\n• Disease identification and treatment\n• Soil health and fertilization\n• Pest management strategies\n• Watering and irrigation\n• Organic farming methods\n• Crop management tips"

    @staticmethod
    def _clean_response(text: str) -> str:
        """Ensure all AI responses are professional, human-readable, and never raw JSON."""
        import json
        import re
        if not text:
            return ""
        text = text.strip()
        # Remove code block markers
        text = text.replace('```json', '').replace('```', '').strip()
        # If the response is JSON, format it professionally
        if text.startswith('{') and text.endswith('}'):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    lines = []
                    for key, value in data.items():
                        pretty_key = key.replace('_', ' ').title()
                        if isinstance(value, dict):
                            lines.append(f"**{pretty_key}:**")
                            for subkey, subval in value.items():
                                lines.append(f"  - {subkey.replace('_', ' ').title()}: {subval}")
                        elif isinstance(value, list):
                            lines.append(f"**{pretty_key}:**")
                            for item in value:
                                lines.append(f"  - {item}")
                        else:
                            lines.append(f"**{pretty_key}:** {value}")
                    return "\n".join(lines)
            except Exception:
                pass  # If JSON parsing fails, fall through to plain text
        # Remove any remaining JSON-like artifacts
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'"[^"]*":\s*"[^"]*"', '', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        # Capitalize first letter, ensure professional tone
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    async def get_disease_recommendations(self, disease_name: str, confidence: float, crop_type: str, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive treatment, prevention, and immediate action recommendations for a disease"""
        
        logger.info(f"Getting disease recommendations for {disease_name} ({confidence:.1%} confidence) on {crop_type}")
        
        # Create cache key
        cache_key = f"{disease_name}_{confidence}_{crop_type}"
        
        # Check cache first
        if cache_key in self._recommendations_cache:
            logger.info(f"Using cached recommendations for {disease_name}")
            return self._recommendations_cache[cache_key]
        
        try:
            # Use the specialized disease analysis prompt
            prompt = self._get_disease_analysis_prompt(disease_name, confidence, crop_type)
            
            # Instead of calling a paid API, just return fallback for now
            fallback = self._get_structured_fallback_recommendations(disease_name, confidence, crop_type)
            self._recommendations_cache[cache_key] = fallback
            return fallback
                
        except Exception as e:
            logger.error(f"Error getting recommendations for {disease_name}: {str(e)}")
            fallback = self._get_structured_fallback_recommendations(disease_name, confidence, crop_type)
            
            # Cache the fallback recommendations
            self._recommendations_cache[cache_key] = fallback
            
            return fallback

    def _get_structured_fallback_recommendations(self, disease_name: str, confidence: float, crop_type: str) -> Dict[str, Any]:
        """Get structured fallback recommendations when LLM is unavailable"""
        
        # Provide specific recommendations for common diseases
        if "cashew_leafminer" in disease_name.lower():
            return {
                "disease_overview": f"{disease_name} is a serious pest that affects cashew trees by mining into the leaves, causing significant damage to the foliage and reducing photosynthesis. The larvae create serpentine mines in the leaves, which can lead to defoliation and reduced nut production.",
                "immediate_actions": "1. Inspect all cashew trees for leafminer damage\n2. Remove and destroy heavily infested leaves\n3. Apply neem oil or insecticidal soap to affected areas\n4. Monitor surrounding trees for spread\n5. Consider introducing natural predators",
                "treatment_protocols": {
                    "organic": "Apply neem oil (2-3% solution) every 7-10 days\nUse Bacillus thuringiensis (Bt) spray\nIntroduce beneficial insects like parasitic wasps\nApply garlic or chili pepper spray as deterrent",
                    "chemical": "Use spinosad-based insecticides\nApply abamectin if severe infestation\nUse systemic insecticides as last resort\nAlways follow label instructions",
                    "application": "Apply treatments early morning or evening\nCover both sides of leaves thoroughly\nRepeat applications every 7-10 days\nAvoid spraying during flowering"
                },
                "prevention": "Plant cashew trees with adequate spacing\nMaintain good air circulation\nUse resistant cashew varieties when available\nPractice regular monitoring\nKeep area clean of fallen leaves\nApply preventive neem treatments",
                "monitoring": "Check leaves weekly for mining damage\nMonitor for new leaf mines\nTrack treatment effectiveness\nDocument infestation levels\nWatch for natural predators",
                "cost_effective": "Use homemade neem oil solutions\nPractice good cultural methods\nJoin local cashew farmer groups\nShare monitoring responsibilities with neighbors\nUse integrated pest management",
                "severity_level": "High",
                "professional_help": "Consult agricultural extension if more than 30% of leaves are affected or if treatments are not working after 2-3 applications"
            }
        elif "leaf" in disease_name.lower() and "miner" in disease_name.lower():
            return {
                "disease_overview": f"{disease_name} is a leaf-mining pest that creates tunnels in plant leaves, reducing photosynthesis and plant health. The larvae feed between the upper and lower leaf surfaces, creating visible serpentine mines.",
                "immediate_actions": "1. Remove and destroy heavily mined leaves\n2. Apply neem oil or insecticidal soap\n3. Monitor for new mines\n4. Consider introducing natural predators\n5. Isolate severely affected plants",
                "treatment_protocols": {
                    "organic": "Apply neem oil (2-3% solution)\nUse Bacillus thuringiensis (Bt)\nIntroduce parasitic wasps\nApply garlic or chili pepper spray",
                    "chemical": "Use spinosad-based products\nApply abamectin if needed\nUse systemic insecticides as last resort",
                    "application": "Apply early morning or evening\nCover both leaf surfaces\nRepeat every 7-10 days\nAvoid flowering periods"
                },
                "prevention": "Maintain good plant spacing\nEnsure proper air circulation\nUse resistant varieties\nRegular monitoring\nClean up fallen leaves",
                "monitoring": "Check leaves weekly for mines\nMonitor treatment effectiveness\nTrack natural predator presence\nDocument damage levels",
                "cost_effective": "Use homemade neem solutions\nPractice cultural controls\nJoin farmer groups\nShare monitoring with neighbors",
                "severity_level": "Moderate",
                "professional_help": "Seek help if more than 25% of leaves are affected or treatments fail after 2-3 applications"
            }
        else:
            return {
                "disease_overview": f"General information about {disease_name} affecting {crop_type} plants. This disease can impact plant health and productivity.",
                "immediate_actions": "1. Isolate affected plants\n2. Remove infected parts\n3. Improve air circulation\n4. Avoid overhead watering\n5. Use clean tools",
                "treatment_protocols": {
                    "organic": "Apply neem oil or copper-based fungicides\nUse beneficial microbes\nImprove soil health\nApply garlic or chili pepper spray",
                    "chemical": "Consult with agricultural extension for chemical options\nUse appropriate fungicides or insecticides\nFollow label instructions carefully",
                    "application": "Apply treatments early morning or evening\nCover all affected areas thoroughly\nRepeat as needed\nAvoid flowering periods"
                },
                "prevention": "Use disease-resistant varieties\nPractice crop rotation\nMaintain proper spacing\nKeep tools clean\nMonitor regularly",
                "monitoring": "Check plants daily for new symptoms\nMonitor treatment effectiveness\nDocument progress\nWatch for natural predators",
                "cost_effective": "Use homemade remedies like baking soda spray\nPractice good cultural methods\nJoin local farming groups\nShare monitoring responsibilities",
                "severity_level": "Moderate",
                "professional_help": "Consult agricultural extension if symptoms worsen or spread rapidly"
            }

    async def chat_with_ai(self, user_message: str, conversation_history: Optional[List[dict]] = None, models: Optional[list] = None) -> str:
        """Chat method using OpenRouter's OpenAI-compatible API for chat completions only, with extra_headers for referer and title."""
        import asyncio
        from openai import OpenAI

        # Use OpenRouter API key and endpoint
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY,
            )
            messages = []
            messages.append({
                "role": "system",
                "content": self._get_agricultural_system_prompt()
            })
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({
                "role": "user",
                "content": user_message
            })
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://phamiq.ai/",
                        "X-Title": "Phamiq AI"
                    },
                    model="openai/gpt-4o",
                    messages=messages,
                )
            )
            content = completion.choices[0].message.content
            return content
        except Exception as e:
            logger.error(f"Error in OpenRouter chat_with_ai: {str(e)}")
            return self._get_fallback_response(user_message)

    async def generate_image(self, prompt: str, models: Optional[List[str]] = None) -> str:
        """Generate image using AlleAI image generation endpoint"""
        
        logger.info("FreeAI image generation returns placeholder image.")
        return "https://placehold.co/400x250/png?text=Image+Generation+Unavailable"

# Global service instance
freeai_service = FreeAIService()