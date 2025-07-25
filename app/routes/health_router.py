from fastapi import APIRouter
from app.models.schemas import HealthResponse, BasicHealthResponse, ClassesResponse
from app.config import settings, CLASS_DICT, IDX_TO_CLASS
from app.services.prediction_service import prediction_service

router = APIRouter(tags=["Health"])

@router.get("/", response_model=BasicHealthResponse)
async def root():
    """Basic health check endpoint"""
    return BasicHealthResponse(
        message="Crop Disease Classification API",
        status="running",
        model_loaded=prediction_service.is_model_loaded()
    )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if prediction_service.is_model_loaded() else "unhealthy",
        model_loaded=prediction_service.is_model_loaded(),
        supported_classes=len(IDX_TO_CLASS),
        classes=list(IDX_TO_CLASS.values())
    )

@router.get("/classes", response_model=ClassesResponse)
async def get_classes():
    """Get all supported disease classes"""
    crops = {
        "cashew": [v for k, v in CLASS_DICT.items() if k.startswith('c')],
        "cassava": [v for k, v in CLASS_DICT.items() if k.startswith('ca')],
        "maize": [v for k, v in CLASS_DICT.items() if k.startswith('m')],
        "tomato": [v for k, v in CLASS_DICT.items() if k.startswith('t')]
    }
    
    return ClassesResponse(
        total_classes=len(IDX_TO_CLASS),
        classes=IDX_TO_CLASS,
        crops=crops
    )