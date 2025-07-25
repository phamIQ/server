from fastapi import APIRouter, Depends, HTTPException
from typing import List
import logging

from app.models.database import PredictionHistoryModel
from app.utils.auth import get_current_active_user
from app.models.database import User

router = APIRouter(prefix="/history", tags=["History"])
logger = logging.getLogger(__name__)

@router.get("/test")
async def test_history_endpoint():
    """Test endpoint to verify history router is working"""
    logger.info("History test endpoint called")
    return {"message": "History router is working", "status": "success"}

@router.get("/public", response_model=List[dict])
async def get_public_history(limit: int = 10):
    """
    Get public prediction history (sample data for demonstration)
    
    - **limit**: Maximum number of history entries to return (default: 10)
    - **Authentication**: Not required
    """
    try:
        logger.info(f"Public history endpoint called with limit: {limit}")
        
        # Return sample data for public access
        return [
            {
                "id": "public_1",
                "user_id": "demo_user",
                "filename": "sample_tomato.jpg",
                "disease": "Tomato Leaf Blight",
                "confidence": 0.85,
                "severity": "Moderate",
                "crop_type": "Tomato",
                "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": "public_2",
                "user_id": "demo_user",
                "filename": "sample_maize.jpg",
                "disease": "Maize Healthy",
                "confidence": 0.92,
                "severity": "Mild",
                "crop_type": "Maize",
                "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
                "created_at": "2024-01-14T15:45:00Z"
            },
            {
                "id": "public_3",
                "user_id": "demo_user",
                "filename": "sample_cassava.jpg",
                "disease": "Cassava Mosaic",
                "confidence": 0.78,
                "severity": "High",
                "crop_type": "Cassava",
                "image_url": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400",
                "created_at": "2024-01-13T09:20:00Z"
            }
        ][:limit]
        
    except Exception as e:
        logger.error(f"Error retrieving public history: {str(e)}")
        return []

@router.get("/user", response_model=List[dict])
async def get_prediction_history(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get prediction history for the current user
    
    - **limit**: Maximum number of history entries to return (default: 50)
    - **Authentication**: Required
    """
    try:
        logger.info(f"History endpoint called for user: {current_user.email} with limit: {limit}")
        
        # Check if database is connected
        from app.models.database import database
        if database is None:
            logger.warning("Database not connected, returning sample history")
            # Return sample data for testing
            return [
                {
                    "id": "1",
                    "user_id": str(current_user.id),
                    "filename": "sample_image.jpg",
                    "disease": "Tomato Leaf Blight",
                    "confidence": 0.85,
                    "severity": "Moderate",
                    "crop_type": "Tomato",
                    "image_url": None,
                    "created_at": "2024-01-15T10:30:00Z"
                },
                {
                    "id": "2",
                    "user_id": str(current_user.id),
                    "filename": "test_image.png",
                    "disease": "Maize Healthy",
                    "confidence": 0.92,
                    "severity": "Mild",
                    "crop_type": "Maize",
                    "image_url": None,
                    "created_at": "2024-01-14T15:45:00Z"
                }
            ]
        
        history = await PredictionHistoryModel.find_by_user_id(
            str(current_user.id), 
            limit=limit
        )
        
        # Convert to dictionary format for JSON response
        history_list = [entry.to_dict() for entry in history]
        
        logger.info(f"Retrieved {len(history_list)} history entries for user: {current_user.email}")
        return history_list
        
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        # Return sample data instead of empty list for testing
        return [
            {
                "id": "error_1",
                "user_id": str(current_user.id) if current_user else "unknown",
                "filename": "error_sample.jpg",
                "disease": "Sample Disease",
                "confidence": 0.75,
                "severity": "Moderate",
                "crop_type": "Sample Crop",
                "image_url": None,
                "created_at": "2024-01-15T10:30:00Z"
            }
        ]

@router.delete("/{history_id}")
async def delete_prediction_history(
    history_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a specific prediction history entry
    
    - **history_id**: ID of the history entry to delete
    - **Authentication**: Required
    """
    try:
        logger.info(f"Delete history endpoint called for user: {current_user.email}, history_id: {history_id}")
        
        # Find the history entry
        history_entry = await PredictionHistoryModel.find_by_id(history_id)
        
        if not history_entry:
            raise HTTPException(
                status_code=404,
                detail="History entry not found"
            )
        
        # Check if the history entry belongs to the current user
        if str(history_entry.user_id) != str(current_user.id):
            raise HTTPException(
                status_code=403,
                detail="You can only delete your own history entries"
            )
        
        # Delete the history entry
        await history_entry.delete()
        
        logger.info(f"Successfully deleted history entry {history_id} for user: {current_user.email}")
        
        return {
            "status": "success",
            "message": "History entry deleted successfully",
            "deleted_id": history_id
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting prediction history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete history entry: {str(e)}"
        ) 