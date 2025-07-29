# Phamiq Backend API Documentation

## Overview

Phamiq is a comprehensive agricultural AI platform that provides crop disease classification, AI-powered recommendations, and multispectral analysis. The backend is built with FastAPI and offers a robust REST API for agricultural professionals and farmers.

## Core Features

### **Crop Disease Classification**
- **AI-Powered Detection**: Uses EfficientNetV2 model to classify 24 different crop diseases
- **Supported Crops**: Cashew, Cassava, Maize, and Tomato
- **Real-time Analysis**: Upload images and get instant disease predictions with confidence scores
- **Comprehensive Results**: Returns top predictions with detailed disease information

### **AI-Powered Recommendations**
- **Free AI Integration**: Uses OpenRouter for cost-effective AI recommendations
- **Disease Treatment Plans**: Get detailed treatment protocols for detected diseases
- **Organic & Chemical Solutions**: Provides both organic and conventional treatment options
- **Prevention Strategies**: Long-term prevention and monitoring recommendations

### **Multispectral Analysis**
- **Satellite Data Processing**: Analyzes multispectral satellite imagery
- **Crop Suitability Assessment**: Determines optimal crop types for specific areas
- **Environmental Indices**: Calculates NDVI and other vegetation indices
- **Background Processing**: Handles large datasets asynchronously

### **User Management**
- **Authentication**: JWT-based authentication system
- **Google OAuth**: Social login integration
- **User Profiles**: Personalized user accounts and settings
- **Prediction History**: Track and manage past analyses

## API Endpoints

### Authentication (`/auth`)
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/google` - Google OAuth login
- `POST /auth/refresh` - Refresh access token

### Disease Prediction (`/predict`)
- `POST /predict/` - Upload image for disease classification
- `GET /predict/recommendations/{disease_name}` - Get AI recommendations for specific disease
- `GET /predict/test-llm` - Test AI integration
- `GET /predict/cache/stats` - Get recommendation cache statistics
- `DELETE /predict/cache` - Clear recommendation cache

### Multispectral Analysis (`/predict/multispectral`)
- `POST /predict/multispectral` - Analyze multispectral data synchronously
- `POST /predict/multispectral/async` - Submit background analysis job
- `GET /predict/multispectral/status/{job_id}` - Check job status

### AI Services (`/ai`)
- `POST /ai/chat` - General AI chat
- `POST /ai/disease-analysis` - Specialized disease analysis
- `POST /ai/generate-title` - Generate content titles
- `POST /ai/generate-description` - Generate content descriptions
- `POST /ai/generate-image` - Generate images (placeholder)
- `GET /ai/status` - Check AI service status
- `GET /ai/test` - Test AI functionality

### User Management (`/user`)
- `GET /user/profile` - Get user profile
- `PUT /user/profile` - Update user profile
- `GET /user/predictions` - Get user's prediction history

### History (`/history`)
- `GET /history/` - Get prediction history
- `DELETE /history/{prediction_id}` - Delete specific prediction
- `GET /history/stats` - Get usage statistics

### Discovery (`/discovery`)
- `GET /discovery/posts` - Get discovery posts
- `POST /discovery/posts` - Create new discovery post
- `GET /discovery/posts/{post_id}` - Get specific post

### Health & Status
- `GET /` - API root and version info
- `GET /health` - Health check endpoint
- `GET /test-ai` - AI service test
- `GET /ai-status` - Detailed AI service status

## Technical Architecture

### **Framework & Dependencies**
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation and settings management
- **Motor**: Async MongoDB driver
- **OpenCV**: Image processing and computer vision
- **ONNX Runtime**: Efficient model inference
- **OpenAI/OpenRouter**: AI service integration

### **Database**
- **MongoDB**: NoSQL database for user data and prediction history
- **Collections**: Users, Predictions, Analysis Jobs, Discovery Posts

### **AI Models**
- **EfficientNetV2**: Pre-trained model for crop disease classification
- **24 Disease Classes**: Covering 4 major crops (Cashew, Cassava, Maize, Tomato)
- **ONNX Format**: Optimized for production deployment

### **Security Features**
- **JWT Authentication**: Secure token-based authentication
- **CORS Protection**: Cross-origin resource sharing configuration
- **Input Validation**: Comprehensive request validation
- **File Upload Limits**: Secure file handling with size restrictions

## Environment Variables

```env
# Database
MONGODB_URL=mongodb://localhost:27017/phamiq
MONGODB_DB_NAME=phamiq

# Authentication
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Services
OPENROUTER_API_KEY=your_openrouter_key

# OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Frontend
FRONTEND_URL=http://localhost:3000
SITE_URL=https://your-domain.com
SITE_NAME=Phamiq

# File Upload
MAX_FILE_SIZE=1073741824  # 1GB
```

## Deployment

### **Docker Deployment**
```bash
# Build the image
docker build -t phamiq-backend .

# Run the container
docker run -p 8000:8000 phamiq-backend
```

### **Render Deployment**
- Connect your GitHub repository to Render
- Set the root directory to `/backend`
- Configure environment variables in Render dashboard
- Deploy automatically on code changes

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, you can access:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Support & Contributing

For technical support or to contribute to the project:
1. Check the API documentation at `/docs`
2. Review the health endpoints for service status
3. Test AI integration using `/test-ai` endpoint

---

**Version**: 1.0.0  
**Framework**: FastAPI  
**Database**: MongoDB  
**AI Integration**: OpenRouter (Free)  
**Deployment**: Docker-ready for Render
