from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional
from datetime import datetime

# Authentication Schemas
class UserBase(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    first_name: str = Field(..., min_length=1, max_length=50, description="User first name")
    last_name: str = Field(..., min_length=1, max_length=50, description="User last name")

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    confirm_password: str = Field(..., description="Password confirmation")

class UserLogin(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in minutes")

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None

# Existing Schemas
class PredictionItem(BaseModel):
    """Single prediction item"""
    class_name: str = Field(..., alias="class", description="Disease class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    confidence_percentage: str = Field(..., description="Confidence as percentage string")
    
    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    status: str = Field(default="success", description="Response status")
    filename: Optional[str] = Field(None, description="Uploaded filename")
    predictions: List[PredictionItem] = Field(..., description="List of predictions")
    total_classes: int = Field(..., description="Total number of classes")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    supported_classes: int = Field(..., description="Number of supported classes")
    classes: List[str] = Field(..., description="List of all class names")

class BasicHealthResponse(BaseModel):
    """Basic health check response"""
    message: str = Field(..., description="Status message")
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")

class ClassesResponse(BaseModel):
    """Classes information response"""
    total_classes: int = Field(..., description="Total number of classes")
    classes: Dict[int, str] = Field(..., description="Index to class mapping")
    crops: Dict[str, List[str]] = Field(..., description="Classes grouped by crop")

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

# Recommendation Schemas
class TreatmentProtocols(BaseModel):
    """Treatment protocols for disease management"""
    organic: str = Field(..., description="Organic treatment methods")
    chemical: str = Field(..., description="Chemical treatment options")
    application: str = Field(..., description="How and when to apply treatments")

class DiseaseRecommendations(BaseModel):
    """Comprehensive disease treatment and prevention recommendations from LLM"""
    disease_overview: str = Field(..., description="Brief disease description and key symptoms")
    immediate_actions: str = Field(..., description="Step-by-step immediate response plan")
    treatment_protocols: TreatmentProtocols = Field(..., description="Treatment protocols")
    prevention: str = Field(..., description="Long-term prevention strategies")
    monitoring: str = Field(..., description="How to monitor progress and effectiveness")
    cost_effective: str = Field(..., description="Budget-friendly solutions")
    severity_level: str = Field(..., description="Disease severity level")
    professional_help: str = Field(..., description="When to consult agricultural experts")

class EnhancedPredictionResponse(BaseModel):
    """Enhanced prediction response with LLM recommendations"""
    status: str = Field(default="success", description="Response status")
    filename: Optional[str] = Field(None, description="Uploaded filename")
    predictions: List[PredictionItem] = Field(..., description="List of predictions")
    total_classes: int = Field(..., description="Total number of classes")
    recommendations: DiseaseRecommendations = Field(..., description="LLM disease recommendations")
    llm_available: bool = Field(..., description="Whether LLM recommendations are available")