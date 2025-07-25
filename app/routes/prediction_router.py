from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
import os
import tempfile
import zipfile
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.models.schemas import PredictionResponse, ErrorResponse, EnhancedPredictionResponse, DiseaseRecommendations
from app.services.prediction_service import prediction_service
from app.services.alleai_service import freeai_service
from app.config import settings, IDX_TO_CLASS
from app.utils.auth import get_current_active_user
from app.models.database import User, PredictionHistoryModel, AnalysisJobModel

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

@router.get("/test-llm")
async def test_llm_endpoint():
    """Test endpoint to verify LLM integration"""
    try:
        if not freeai_service.is_available():
            return {
                "status": "warning",
                "message": "AI service not available - API key not configured",
                "llm_available": False
            }
        
        # Test with a simple disease
        recommendations = await freeai_service.get_disease_recommendations(
            "Tomato Leaf Blight", 
            0.85, 
            "Tomato"
        )
        
        return {
            "status": "success",
            "message": "LLM integration working",
            "llm_available": True,
            "sample_recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"LLM test error: {str(e)}")
        return {
            "status": "error",
            "message": f"LLM test failed: {str(e)}",
            "llm_available": False
        }

@router.get("/recommendations/{disease_name}")
async def get_disease_recommendations(
    disease_name: str,
    confidence: float = Query(0.8, ge=0.0, le=1.0, description="Confidence level"),
    crop_type: str = Query("Unknown", description="Crop type"),
    models: Optional[List[str]] = Query(None, description="AI models to use"),
    current_user: User = Depends(get_current_active_user)
):
    """Get LLM recommendations for a specific disease"""
    try:
        if not freeai_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service is required but not available. Please configure AI service API key."
            )
        
        # Get comprehensive LLM recommendations
        recommendations_data = await freeai_service.get_disease_recommendations(
            disease_name, 
            confidence, 
            crop_type,
            models
        )
        
        return {
            "status": "success",
            "disease_name": disease_name,
            "confidence": confidence,
            "crop_type": crop_type,
            "recommendations": recommendations_data,
            "llm_available": True
        }
        
    except Exception as e:
        logger.error(f"Error getting disease recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate disease recommendations: {str(e)}"
        )

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = freeai_service.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@router.delete("/cache")
async def clear_cache():
    """Clear the recommendations cache"""
    try:
        freeai_service.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.post("/", response_model=EnhancedPredictionResponse)
async def predict_disease(
    file: UploadFile = File(..., description="Image file (jpg, jpeg, png)"),
    top_k: int = Query(3, ge=1, le=len(IDX_TO_CLASS), description="Number of top predictions"),
    models: Optional[List[str]] = Query(None, description="AI models to use for recommendations"),
    current_user: User = Depends(get_current_active_user)
):
    """Predict disease from uploaded image with enhanced LLM recommendations"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        image_data = await file.read()
        if len(image_data) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        
        # Get predictions
        predictions = prediction_service.predict_disease_from_image(image, top_k)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to get predictions")
        
        # Get LLM recommendations for top prediction - LLM is now required
        if not freeai_service.is_available():
            logger.error("AI service not available - LLM is required for disease analysis")
            raise HTTPException(
                status_code=503,
                detail="LLM service is required for disease analysis but not available. Please configure AI service API key."
            )
        
        if predictions:
            top_prediction = predictions[0]
            
            try:
                # Extract crop type from disease name
                disease_name = top_prediction.class_name
                crop_type = "Unknown Crop"
                if disease_name:
                    lower_disease = disease_name.lower()
                    if "cashew" in lower_disease:
                        crop_type = "Cashew"
                    elif "cassava" in lower_disease:
                        crop_type = "Cassava"
                    elif "maize" in lower_disease:
                        crop_type = "Maize"
                    elif "tomato" in lower_disease:
                        crop_type = "Tomato"
                
                # Get comprehensive LLM recommendations - this is now mandatory
                recommendations_data = await freeai_service.get_disease_recommendations(
                    disease_name, 
                    top_prediction.confidence, 
                    crop_type,
                    models
                )
                recommendations = DiseaseRecommendations(**recommendations_data)
                logger.info(f"Generated comprehensive LLM recommendations for {disease_name}")
                
            except Exception as e:
                logger.error(f"Error getting LLM recommendations: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate disease recommendations: {str(e)}"
                )
        
        # Save to history
        try:
            # Extract the top prediction for history
            top_prediction = predictions[0]
            disease_name = top_prediction.class_name
            confidence = top_prediction.confidence
            
            # Determine severity based on confidence
            severity = 'Mild'
            if confidence > 0.8:
                severity = 'Severe'
            elif confidence > 0.6:
                severity = 'Moderate'
            
            # Extract crop type from disease name
            crop_type = "Unknown Crop"
            if disease_name:
                lower_disease = disease_name.lower()
                if "cashew" in lower_disease:
                    crop_type = "Cashew"
                elif "cassava" in lower_disease:
                    crop_type = "Cassava"
                elif "maize" in lower_disease:
                    crop_type = "Maize"
                elif "tomato" in lower_disease:
                    crop_type = "Tomato"
            
            await PredictionHistoryModel.create(
                user_id=str(current_user.id),
                filename=file.filename,
                disease=disease_name,
                confidence=confidence,
                severity=severity,
                crop_type=crop_type,
                image_url=None,  # We don't store the actual image URL in history
                recommendations=recommendations_data if 'recommendations_data' in locals() else None
            )
            logger.info(f"Saved prediction history for user {current_user.email}")
        except Exception as e:
            logger.error(f"Failed to save prediction history: {str(e)}")
            # Don't fail the request if history saving fails
        
        return EnhancedPredictionResponse(
            success=True,
            predictions=predictions,
            total_classes=len(IDX_TO_CLASS),
            recommendations=recommendations if 'recommendations' in locals() else None,
            llm_available=True
        )
    
    except HTTPException as e:
        logger.error(f"Prediction error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/multispectral/async", response_model=Dict[str, Any])
async def submit_multispectral_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multispectral data files (.txt, .zip, or band files)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit a multispectral analysis job (async).
    Returns a job_id immediately. Use /multispectral/status/{job_id} to check status/result.
    """
    # Create job in DB
    job = await AnalysisJobModel.create(user_id=str(current_user.id))
    job_id = str(job.id)
    # Save files to temp dir for background processing
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for upload in files:
        file_path = os.path.join(temp_dir, upload.filename)
        contents = await upload.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        file_paths.append(file_path)
    # Launch background task
    background_tasks.add_task(process_multispectral_job, job_id, file_paths)
    return {"job_id": job_id, "status": "pending"}

async def process_multispectral_job(job_id: str, file_paths: list):
    from app.services.multispectral_service import MultispectralAnalyzer
    import os
    import traceback
    try:
        job = await AnalysisJobModel.find_by_id(job_id)
        if not job:
            return
        await job.update_status("processing")
        # Find MTL file
        mtl_path = None
        for path in file_paths:
            if path.lower().endswith('.txt') and '_mtl' in path.lower():
                mtl_path = path
                break
        if not mtl_path:
            await job.update_status("failed", error="No MTL .txt metadata file found among uploads.")
            return
        analyzer = MultispectralAnalyzer()
        results = await analyzer.analyze_from_mtl(mtl_path)
        # Save to history (tag as multispectral)
        try:
            from app.models.database import PredictionHistoryModel
            # You may want to extract user_id from job
            user_id = str(job.user_id)
            await PredictionHistoryModel.create(
                user_id=user_id,
                filename=os.path.basename(mtl_path),
                disease=results.get('prediction', 'Multispectral Analysis'),
                confidence=results.get('confidence', 0),
                severity=results.get('severity', 'N/A'),
                crop_type=results.get('best_crop', 'N/A'),
                image_url=None,
                is_multispectral=True
            )
        except Exception as hist_err:
            print(f"Failed to save multispectral result to history: {hist_err}")
        await job.update_status("completed", result={
            "status": results.get("status", "success"),
            "filename": mtl_path,
            "analysis_type": "multispectral",
            "results": results
        })
    except Exception as e:
        tb = traceback.format_exc()
        job = await AnalysisJobModel.find_by_id(job_id)
        if job:
            await job.update_status("failed", error=f"Analysis failed: {str(e)}\n{tb}")
    finally:
        # Clean up temp files
        import shutil
        temp_dir = os.path.dirname(file_paths[0]) if file_paths else None
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@router.get("/multispectral/status/{job_id}", response_model=Dict[str, Any])
async def get_multispectral_job_status(job_id: str, current_user: User = Depends(get_current_active_user)):
    job = await AnalysisJobModel.find_by_id(job_id)
    if not job or str(job.user_id) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

@router.post("/multispectral", response_model=Dict[str, Any])
async def analyze_multispectral(
    files: List[UploadFile] = File(..., description="Multispectral data files (.txt, .zip, and/or band files)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze multispectral satellite data for crop suitability and environmental indices
    
    - **files**: List of multispectral data files (.txt, .zip, and/or band files)
    - **Authentication**: Required
    """
    try:
        # Accept both single and multiple file uploads for backward compatibility
        if not isinstance(files, list):
            files = [files]
        if not files or not all(f.filename for f in files):
            raise HTTPException(
                status_code=400,
                detail="At least one file must be uploaded (.txt, .zip, or band files)"
            )
        # Check file sizes
        total_size = 0
        for f in files:
            contents = await f.read()
            total_size += len(contents)
            f.file.seek(0)
        if total_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Total upload too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            mtl_path = None
            # If a .zip is present, process as before
            zip_file = next((f for f in files if f.filename.lower().endswith('.zip')), None)
            if zip_file:
                zip_path = os.path.join(temp_dir, zip_file.filename)
                contents = await zip_file.read()
                with open(zip_path, 'wb') as f:
                    f.write(contents)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                # Find the .txt file (metadata)
                for root, dirs, filelist in os.walk(temp_dir):
                    for fname in filelist:
                        if fname.lower().endswith('.txt') and '_mtl' in fname.lower():
                            mtl_path = os.path.join(root, fname)
                            break
                    if mtl_path:
                        break
                if not mtl_path:
                    raise HTTPException(status_code=400, detail="No MTL .txt metadata file found in zip.")
            else:
                # Save all files to temp_dir
                for upload in files:
                    file_path = os.path.join(temp_dir, upload.filename)
                    contents = await upload.read()
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    if upload.filename.lower().endswith('.txt') and '_mtl' in upload.filename.lower():
                        mtl_path = file_path
                if not mtl_path:
                    raise HTTPException(status_code=400, detail="No MTL .txt metadata file found among uploads.")
            # Import multispectral analysis functions
            try:
                from app.services.multispectral_service import MultispectralAnalyzer
                analyzer = MultispectralAnalyzer()
                results = await analyzer.analyze_from_mtl(mtl_path)
                return {
                    "status": results.get("status", "success"),
                    "filename": mtl_path,
                    "analysis_type": "multispectral",
                    "results": results
                }
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="Multispectral analysis service not available"
                )
            except Exception as e:
                logger.error(f"Multispectral analysis error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing multispectral request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )