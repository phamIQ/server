from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
# from app.models.database import User  # REMOVE to break circular import
from app.utils.passwords import get_password_hash, verify_password

if TYPE_CHECKING:
    from app.models.database import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        user_id = payload.get("user_id")
        if email is None or user_id is None:
            return None
        return {"email": email, "user_id": user_id}
    except JWTError:
        return None

# For type hints, use 'User' as a string to avoid circular import
async def authenticate_user(email: str, password: str) -> Optional['User']:
    """Authenticate a user with email and password"""
    from app.models.database import User  # Import here to avoid circular import
    
    user = await User.find_by_email(email)
    if not user:
        return None
    if not user.verify_password(password):
        return None
    return user

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> 'User':
    """Get the current authenticated user"""
    from app.models.database import User  # Import here to avoid circular import
    
    token = credentials.credentials
    print(f"Received token: {token[:50]}...")
    
    payload = verify_token(token)
    if payload is None:
        print("Token verification failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("email")
    user_id = payload.get("user_id")
    print(f"Token payload - email: {email}, user_id: {user_id}")
    
    if email is None or user_id is None:
        print("Missing email or user_id in token payload")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await User.find_by_id(user_id)
    if user is None:
        print(f"User not found for ID: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.email != email:
        print(f"Email mismatch: token={email}, user={user.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"User authenticated successfully: {user.email}")
    return user

async def get_current_active_user(current_user: 'User' = Depends(get_current_user)) -> 'User':
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user 