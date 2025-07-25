from fastapi import APIRouter, Depends, HTTPException, status, Body, Path
from app.utils.auth import get_current_active_user
from app.models.database import User
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random
from uuid import uuid4

router = APIRouter(prefix="/discovery", tags=["Discovery"])

# Pydantic models for discovery content
class TrendingDisease(BaseModel):
    id: str
    disease_name: str
    crop_type: str
    trend: str  # "increasing", "stable", "decreasing"
    cases_count: int
    description: str
    severity: str  # "low", "medium", "high"
    region: str
    last_updated: datetime
    image_url: Optional[str] = None

class Insight(BaseModel):
    id: str
    title: str
    description: str
    category: str  # "research", "community", "education", "tips"
    author: str
    published_date: datetime
    read_time: int  # minutes
    image_url: Optional[str] = None
    tags: List[str] = []

class DiseaseAlert(BaseModel):
    id: str
    disease_name: str
    crop_type: str
    region: str
    alert_level: str  # "low", "medium", "high", "critical"
    description: str
    recommendations: List[str]
    issued_date: datetime

class InsightCreate(BaseModel):
    title: str
    description: str
    category: str
    author: str
    image_url: Optional[str] = None
    tags: List[str] = []

# Mock data for trending diseases
TRENDING_DISEASES_DATA = [
    {
        "id": "1",
        "disease_name": "Late Blight",
        "crop_type": "Potato",
        "trend": "increasing",
        "cases_count": 245,
        "description": "Common fungal disease affecting potato crops worldwide, especially during wet conditions",
        "severity": "high",
        "region": "Global",
        "last_updated": datetime.utcnow() - timedelta(hours=2),
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
    },
    {
        "id": "2",
        "disease_name": "Stem Rust",
        "crop_type": "Wheat",
        "trend": "stable",
        "cases_count": 189,
        "description": "Fungal disease that can cause significant yield losses in wheat crops",
        "severity": "medium",
        "region": "Africa",
        "last_updated": datetime.utcnow() - timedelta(hours=4),
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
    },
    {
        "id": "3",
        "disease_name": "Mosaic Virus",
        "crop_type": "Cassava",
        "trend": "decreasing",
        "cases_count": 156,
        "description": "Viral infection transmitted by whitefly vectors, affecting cassava production",
        "severity": "medium",
        "region": "South America",
        "last_updated": datetime.utcnow() - timedelta(hours=6),
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
    },
    {
        "id": "4",
        "disease_name": "Bacterial Blight",
        "crop_type": "Rice",
        "trend": "increasing",
        "cases_count": 312,
        "description": "Bacterial disease causing significant damage to rice crops in tropical regions",
        "severity": "high",
        "region": "Asia",
        "last_updated": datetime.utcnow() - timedelta(hours=1),
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
    },
    {
        "id": "5",
        "disease_name": "Powdery Mildew",
        "crop_type": "Grape",
        "trend": "stable",
        "cases_count": 98,
        "description": "Fungal disease affecting grapevines, common in humid conditions",
        "severity": "medium",
        "region": "Europe",
        "last_updated": datetime.utcnow() - timedelta(hours=8),
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400"
    }
]

# Mock data for insights
INSIGHTS_DATA = [
    {
        "id": "1",
        "title": "Seasonal Disease Patterns: Understanding Weather Impact on Crop Health",
        "description": "Comprehensive analysis of how weather patterns affect crop disease prevalence and prevention strategies",
        "category": "research",
        "content": "Climate change is significantly impacting agricultural disease patterns worldwide. This research examines the correlation between weather conditions and disease outbreaks, providing farmers with predictive tools for better crop management.",
        "author": "Dr. Sarah Johnson",
        "published_date": datetime.utcnow() - timedelta(days=2),
        "read_time": 8,
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
        "tags": ["climate", "research", "prevention"]
    },
    {
        "id": "2",
        "title": "Community Reports: Regional Disease Outbreaks and Farmer Experiences",
        "description": "Real-time reports from farmers across different regions sharing their experiences with crop diseases",
        "category": "community",
        "content": "This community-driven report compiles experiences from farmers across different regions, providing valuable insights into local disease patterns and successful treatment methods.",
        "author": "Agricultural Community",
        "published_date": datetime.utcnow() - timedelta(days=1),
        "read_time": 5,
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
        "tags": ["community", "reports", "local"]
    },
    {
        "id": "3",
        "title": "Prevention Guides: Comprehensive Strategies for Common Crop Diseases",
        "description": "Expert guides for preventing the most common crop diseases with practical, implementable strategies",
        "category": "education",
        "content": "Prevention is always better than cure. This comprehensive guide covers prevention strategies for the most common crop diseases, including cultural practices, biological controls, and chemical management.",
        "author": "Dr. Michael Chen",
        "published_date": datetime.utcnow() - timedelta(days=3),
        "read_time": 12,
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
        "tags": ["prevention", "guide", "education"]
    },
    {
        "id": "4",
        "title": "Treatment Tips: Expert Advice for Effective Disease Management",
        "description": "Professional treatment recommendations and best practices for managing detected crop diseases",
        "category": "tips",
        "content": "When diseases are detected, quick and effective treatment is crucial. This guide provides expert advice on treatment methods, timing, and follow-up care for various crop diseases.",
        "author": "Dr. Emily Rodriguez",
        "published_date": datetime.utcnow() - timedelta(days=4),
        "read_time": 6,
        "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
        "tags": ["treatment", "tips", "management"]
    }
]

# Mock data for disease alerts
DISEASE_ALERTS_DATA = [
    {
        "id": "1",
        "disease_name": "Late Blight",
        "crop_type": "Potato",
        "region": "North America",
        "alert_level": "high",
        "description": "High humidity and rainfall conditions are favorable for late blight development",
        "recommendations": [
            "Monitor fields daily for early symptoms",
            "Apply preventive fungicides if conditions persist",
            "Remove infected plants immediately",
            "Improve field drainage"
        ],
        "issued_date": datetime.utcnow() - timedelta(hours=1)
    },
    {
        "id": "2",
        "disease_name": "Stem Rust",
        "crop_type": "Wheat",
        "region": "East Africa",
        "alert_level": "medium",
        "description": "Moderate risk of stem rust development in wheat-growing regions",
        "recommendations": [
            "Plant resistant varieties where possible",
            "Monitor for early symptoms",
            "Prepare fungicide applications",
            "Coordinate with local agricultural extension"
        ],
        "issued_date": datetime.utcnow() - timedelta(hours=3)
    }
]

@router.get("/")
async def get_discovery_info():
    """Get discovery service information and available endpoints"""
    try:
        return {
            "status": "success",
            "service": "Phamiq Discovery",
            "endpoints": {
                "trending": "/discovery/trending",
                "insights": "/discovery/insights", 
                "alerts": "/discovery/alerts"
            },
            "message": "Discovery service is ready"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get discovery info: {str(e)}"
        )

@router.get("/trending", response_model=List[TrendingDisease])
async def get_trending_diseases(
    limit: int = 10,
    region: Optional[str] = None,
    # current_user: User = Depends(get_current_active_user)  # Temporarily disable auth for testing
):
    """Get trending diseases with optional region filtering"""
    diseases = TRENDING_DISEASES_DATA.copy()
    
    if region:
        diseases = [d for d in diseases if d["region"].lower() == region.lower()]
    
    # Sort by cases count and return limited results
    diseases.sort(key=lambda x: x["cases_count"], reverse=True)
    return diseases[:limit]

@router.get("/insights", response_model=List[Insight])
async def get_insights(
    category: Optional[str] = None,
    limit: int = 10,
    # current_user: User = Depends(get_current_active_user)  # Temporarily disable auth for testing
):
    """Get agricultural insights and educational content"""
    insights = INSIGHTS_DATA.copy()
    
    if category:
        insights = [i for i in insights if i["category"].lower() == category.lower()]
    
    # Sort by published date and return limited results
    insights.sort(key=lambda x: x["published_date"], reverse=True)
    return insights[:limit]

@router.post("/insights", response_model=Insight)
async def create_insight(
    insight: InsightCreate = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    new_insight = {
        "id": str(uuid4()),
        "title": insight.title,
        "description": insight.description,
        "category": insight.category,
        "author": insight.author,
        "published_date": datetime.utcnow(),
        "read_time": 2,
        "image_url": insight.image_url,
        "tags": insight.tags or [],
    }
    INSIGHTS_DATA.insert(0, new_insight)
    return new_insight

@router.delete("/insights/{insight_id}")
async def delete_insight(
    insight_id: str = Path(...),
    current_user: User = Depends(get_current_active_user)
):
    global INSIGHTS_DATA
    before_count = len(INSIGHTS_DATA)
    INSIGHTS_DATA = [i for i in INSIGHTS_DATA if i["id"] != insight_id]
    if len(INSIGHTS_DATA) == before_count:
        raise HTTPException(status_code=404, detail="Insight not found")
    return {"status": "success", "message": "Insight deleted", "deleted_id": insight_id}

@router.get("/alerts", response_model=List[DiseaseAlert])
async def get_disease_alerts(
    alert_level: Optional[str] = None,
    region: Optional[str] = None,
    # current_user: User = Depends(get_current_active_user)  # Temporarily disable auth for testing
):
    """Get current disease alerts and warnings"""
    alerts = DISEASE_ALERTS_DATA.copy()
    
    if alert_level:
        alerts = [a for a in alerts if a["alert_level"].lower() == alert_level.lower()]
    
    if region:
        alerts = [a for a in alerts if a["region"].lower() == region.lower()]
    
    # Sort by issued date
    alerts.sort(key=lambda x: x["issued_date"], reverse=True)
    return alerts

@router.get("/insights/{insight_id}", response_model=Insight)
async def get_insight_detail(
    insight_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed content for a specific insight"""
    insight = next((i for i in INSIGHTS_DATA if i["id"] == insight_id), None)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return insight

@router.get("/diseases/{disease_id}", response_model=TrendingDisease)
async def get_disease_detail(
    disease_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed information about a specific disease"""
    disease = next((d for d in TRENDING_DISEASES_DATA if d["id"] == disease_id), None)
    if not disease:
        raise HTTPException(status_code=404, detail="Disease not found")
    return disease 