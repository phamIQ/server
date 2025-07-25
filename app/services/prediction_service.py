import onnxruntime
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import logging
from typing import List, Dict, Any, Optional
import os
import time

from app.config import settings, IDX_TO_CLASS
from app.models.schemas import PredictionItem

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Service for handling model predictions with robust caching and error handling.
    
    This service implements a singleton pattern with automatic model loading,
    retry mechanisms, and validation to ensure reliable model inference.
    
    Features:
    - Auto-loading model on initialization
    - Retry mechanism for failed model loads
    - Model validation to ensure working state
    - Comprehensive error handling and logging
    - Memory-efficient caching strategy
    """
    
    def __init__(self, auto_load: bool = True, max_retries: int = 3):
        """
        Initialize the prediction service with optional auto-loading.
        
        Args:
            auto_load (bool): Whether to automatically load the model on initialization
            max_retries (int): Maximum number of retry attempts for model loading
        """
        self.ort_session: Optional[onnxruntime.InferenceSession] = None
        self.transforms = self._create_transforms()
        self.max_retries = max_retries
        self.model_loaded = False
        self.last_load_attempt = None
        
        # Auto-load model if requested (default behavior)
        if auto_load:
            self.load_model()
    
    def _create_transforms(self) -> A.Compose:
        """
        Create image transformation pipeline for model preprocessing.
        
        This pipeline includes:
        - Data augmentation for robustness
        - Normalization to match training data
        - Tensor conversion for ONNX compatibility
        
        Returns:
            A.Compose: Albumentations transformation pipeline
        """
        return A.Compose([
            # Data augmentation for better generalization
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=360, 
                p=0.5
            ),
            # Normalize to ImageNet statistics (standard for most pre-trained models)
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
            # Convert to PyTorch tensor format
            ToTensorV2(),
        ])
    
    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the ONNX model with retry mechanism and validation.
        
        This method implements a robust loading strategy:
        1. Check if model is already loaded (unless force_reload is True)
        2. Attempt loading with retry mechanism
        3. Validate the loaded model
        4. Update loading status and timestamps
        
        Args:
            force_reload (bool): Force reload even if model is already loaded
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails after all retries
            ValueError: If loaded model is invalid
        """
        # Skip loading if model is already loaded and force_reload is False
        if self.model_loaded and not force_reload:
            logger.info("Model already loaded, skipping reload")
            return
        
        logger.info(f"Starting model loading process (max retries: {self.max_retries})")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Model loading attempt {attempt + 1}/{self.max_retries}")
                
                # Check if model file exists
                if not os.path.exists(settings.MODEL_PATH):
                    raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
                
                # Load the ONNX model
                self.ort_session = onnxruntime.InferenceSession(settings.MODEL_PATH)
                logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
                
                # Log model details for debugging
                if self.ort_session is not None:
                    input_details = self.ort_session.get_inputs()[0]
                    logger.info(f"Model input shape: {input_details.shape}")
                    logger.info(f"Model input type: {input_details.type}")
                
                # Validate the loaded model
                if not self.validate_model():
                    raise ValueError("Model validation failed - loaded model is not working correctly")
                
                # Update status
                self.model_loaded = True
                self.last_load_attempt = time.time()
                logger.info("Model loading completed successfully")
                break
                
            except Exception as e:
                logger.error(f"Model loading attempt {attempt + 1} failed: {str(e)}")
                
                # If this is the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    self.model_loaded = False
                    raise RuntimeError(f"Failed to load model after {self.max_retries} attempts: {str(e)}")
                
                # Wait before retrying (exponential backoff)
                wait_time = 2 ** attempt
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def validate_model(self) -> bool:
        """
        Validate that the loaded model is working correctly.
        
        This method performs a simple test inference to ensure:
        - Model can accept input
        - Model can produce output
        - Output format is as expected
        
        Returns:
            bool: True if model validation passes, False otherwise
        """
        try:
            if not self.ort_session:
                logger.error("Model validation failed: No model session available")
                return False
            
            # Create a dummy input for validation
            dummy_input = np.random.randn(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE).astype(np.float32)
            
            # Get model input details
            input_details = self.ort_session.get_inputs()[0]
            input_name = input_details.name
            
            # Run a test inference
            test_output = self.ort_session.run(None, {input_name: dummy_input})
            
            # Validate output format
            if not test_output or len(test_output) == 0:
                logger.error("Model validation failed: No output produced")
                return False
            
            # Check output shape (should match number of classes)
            output_shape = test_output[0].shape
            expected_classes = len(IDX_TO_CLASS)
            
            if output_shape[1] != expected_classes:
                logger.error(f"Model validation failed: Expected {expected_classes} classes, got {output_shape[1]}")
                return False
            
            logger.info("Model validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and validated.
        
        Returns:
            bool: True if model is loaded and working, False otherwise
        """
        return self.model_loaded and self.ort_session is not None
    
    def reload_model_if_needed(self) -> None:
        """
        Reload the model if it's not loaded or if it's been too long since last load.
        
        This method implements a smart reload strategy:
        - Reload if model is not loaded
        - Reload if model validation fails
        - Optionally reload after a certain time period (for long-running services)
        """
        if not self.is_model_loaded():
            logger.info("Model not loaded, attempting to load...")
            self.load_model()
            return
        
        # Optional: Reload if model is old (e.g., > 24 hours)
        # This can be useful for long-running services to handle potential memory issues
        if self.last_load_attempt and (time.time() - self.last_load_attempt) > 86400:  # 24 hours
            logger.info("Model is old, reloading for freshness...")
            self.load_model(force_reload=True)
            return
        
        # Validate current model
        if not self.validate_model():
            logger.warning("Model validation failed, reloading...")
            self.load_model(force_reload=True)
    
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        This method handles various image formats and applies the transformation pipeline:
        1. Convert to RGB format if needed
        2. Resize to model input size
        3. Apply data augmentation and normalization
        4. Convert to tensor format
        
        Args:
            image_array (np.ndarray): Input image as numpy array
            
        Returns:
            np.ndarray: Preprocessed image ready for model inference
            
        Raises:
            ValueError: If image preprocessing fails
        """
        try:
            # Convert to RGB if needed
            if len(image_array.shape) == 2:
                # Grayscale to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Already RGB from PIL
                pass
            else:
                # Assume BGR (OpenCV default) to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image_resized = cv2.resize(image_array, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            
            # Apply transforms (augmentation, normalization, tensor conversion)
            transformed = self.transforms(image=image_resized)
            return transformed["image"].unsqueeze(0).numpy()
        
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def predict(self, image_array: np.ndarray, top_k: int = 3) -> List[PredictionItem]:
        """
        Make prediction on image with robust error handling.
        
        This method ensures the model is loaded and working before making predictions:
        1. Check and reload model if needed
        2. Preprocess the input image
        3. Run inference with error handling
        4. Process and validate outputs
        5. Return top-k predictions
        
        Args:
            image_array (np.ndarray): Input image as numpy array
            top_k (int): Number of top predictions to return
            
        Returns:
            List[PredictionItem]: List of top-k predictions with confidence scores
            
        Raises:
            RuntimeError: If model is not available or inference fails
            ValueError: If prediction processing fails
        """
        try:
            # Ensure model is loaded and working
            self.reload_model_if_needed()
            
            if not self.is_model_loaded():
                raise RuntimeError("Model not loaded and cannot be loaded")
            
            # Preprocess image
            input_data = self.preprocess_image(image_array)
            
            # Run inference safely with comprehensive error checking
            if self.ort_session is None:
                raise RuntimeError("ONNX Runtime session is not initialized.")
            
            if not hasattr(self.ort_session, "get_inputs") or not hasattr(self.ort_session, "run"):
                raise RuntimeError("ONNX Runtime session does not have required methods.")

            ort_inputs_list = self.ort_session.get_inputs()
            if not ort_inputs_list or not hasattr(ort_inputs_list[0], "name"):
                raise RuntimeError("ONNX model input details are missing or malformed.")

            # Prepare inputs and run inference
            ort_inputs = {ort_inputs_list[0].name: input_data}
            ort_outs = self.ort_session.run(None, ort_inputs)

            # Process outputs
            outputs = ort_outs[0]
            
            # Apply softmax to get probabilities
            probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
            probs = probs[0]
            
            # Get top-k predictions
            top_k_idx = np.argsort(probs)[-top_k:][::-1]
            
            predictions = []
            for i in top_k_idx:
                # Validate that the index exists in IDX_TO_CLASS
                if i not in IDX_TO_CLASS:
                    logger.error(f"Invalid class index: {i}, max index: {len(IDX_TO_CLASS)-1}")
                    continue
                
                class_name = IDX_TO_CLASS[i]
                confidence = float(probs[i])
                
                # Validate class name
                if not class_name or not isinstance(class_name, str):
                    logger.error(f"Invalid class name for index {i}: {class_name}")
                    continue
                
                predictions.append(PredictionItem(
                    class_name=class_name,  # Field name is class_name, but serializes to 'class'
                    confidence=confidence,
                    confidence_percentage=f"{confidence:.2%}"
                ))
            
            logger.info(f"Successfully generated {len(predictions)} predictions")
            return predictions
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_disease_from_image(self, image: Image.Image, top_k: int = 3) -> List[PredictionItem]:
        """
        Predict disease from PIL Image object.
        
        This is a convenience method that converts PIL Image to numpy array
        and calls the main predict method.
        
        Args:
            image (Image.Image): PIL Image object
            top_k (int): Number of top predictions to return
            
        Returns:
            List[PredictionItem]: List of top-k predictions with confidence scores
            
        Raises:
            RuntimeError: If model is not available or inference fails
            ValueError: If prediction processing fails
        """
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            
            # Call the main predict method
            return self.predict(image_array, top_k)
            
        except Exception as e:
            logger.error(f"Error in predict_disease_from_image: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Global service instance with auto-loading enabled
prediction_service = PredictionService(auto_load=True, max_retries=3)