from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from datetime import timedelta
from app.models.database import User
from app.models.schemas import UserCreate, UserLogin, UserResponse, Token
from app.utils.auth import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    get_current_active_user
)
from app.config import settings
from authlib.integrations.starlette_client import OAuth
import os

router = APIRouter(tags=["authentication"])

# --- Google OAuth2 Setup ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", getattr(settings, "GOOGLE_CLIENT_ID", None))
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", getattr(settings, "GOOGLE_CLIENT_SECRET", None))
FRONTEND_URL = os.getenv("FRONTEND_URL", getattr(settings, "FRONTEND_URL", None))

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
    }
)

@router.post("/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if passwords match
    if user_data.password != user_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Check if user already exists
    existing_user = await User.find_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    try:
        db_user = await User.create(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            hashed_password=hashed_password
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email, "user_id": str(db_user.id)},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES
    }

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return access token"""
    user = await authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user.to_dict()

@router.get("/verify")
async def verify_token_endpoint(current_user: User = Depends(get_current_active_user)):
    """Verify if the current token is valid"""
    return {"valid": True, "user_id": str(current_user.id), "email": current_user.email}

@router.get("/google/login")
async def google_login(request: Request):
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/google/callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        print("Google OAuth token:", token)  # Debug
        user_info = None
        if "id_token" in token:
            try:
                user_info = await oauth.google.parse_id_token(request, token)
                print("User info from id_token:", user_info)  # Debug
            except Exception as e:
                print("Error parsing id_token:", e)
        if not user_info:
            try:
                resp = await oauth.google.get('userinfo', token=token)
                user_info = await resp.json()
                print("User info from userinfo endpoint:", user_info)  # Debug
            except Exception as e:
                print("Error fetching userinfo:", e)
        if not user_info or not user_info.get('email'):
            print("No user info or email found after all fallbacks.")
            raise HTTPException(status_code=400, detail="Google account email not found.")
        email = user_info.get('email')
        first_name = user_info.get('given_name', '')
        last_name = user_info.get('family_name', '')
        # Check if user exists
        user = await User.find_by_email(email)
        if not user:
            # Create user with unusable password
            user = await User.create(
                email=email,
                first_name=first_name,
                last_name=last_name,
                hashed_password=get_password_hash(os.urandom(16).hex())
            )
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "user_id": str(user.id)},
            expires_delta=access_token_expires
        )
        # Redirect to frontend with token in URL
        redirect_url = f"{FRONTEND_URL}/oauth-callback?token={access_token}"
        return RedirectResponse(redirect_url)
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return RedirectResponse(f"{FRONTEND_URL}/login?error=google_oauth_failed") 