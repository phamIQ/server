from fastapi import APIRouter, Depends, HTTPException, status, Response
from app.models.database import User
from app.utils.auth import get_current_active_user
from pydantic import BaseModel, EmailStr
from typing import Optional
import json

router = APIRouter(prefix="/users", tags=["Users"])

# --- Pydantic Schemas ---
class UserProfileResponse(BaseModel):
    id: str
    email: EmailStr
    first_name: Optional[str]
    last_name: Optional[str]
    location: Optional[str]
    language: Optional[str]
    timezone: Optional[str]

class UpdateProfileRequest(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[EmailStr]
    location: Optional[str]
    language: Optional[str]
    timezone: Optional[str]

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class NotificationSettingsRequest(BaseModel):
    analysis_results: Optional[bool]
    disease_alerts: Optional[bool]

class LanguageSettingsRequest(BaseModel):
    language: str
    timezone: str

# --- Endpoints ---
@router.get("/me", response_model=UserProfileResponse)
async def get_profile(current_user: User = Depends(get_current_active_user)):
    return UserProfileResponse(
        id=str(current_user.id),
        email=current_user.email,
        first_name=getattr(current_user, 'first_name', None),
        last_name=getattr(current_user, 'last_name', None),
        location=getattr(current_user, 'location', None),
        language=getattr(current_user, 'language', None),
        timezone=getattr(current_user, 'timezone', None),
    )

@router.put("/me", response_model=UserProfileResponse)
async def update_profile(data: UpdateProfileRequest, current_user: User = Depends(get_current_active_user)):
    await current_user.update_profile(**data.dict(exclude_unset=True))
    return await get_profile(current_user)

@router.post("/me/change-password")
async def change_password(data: ChangePasswordRequest, current_user: User = Depends(get_current_active_user)):
    if not current_user.hashed_password or not current_user.verify_password(data.old_password):
        raise HTTPException(status_code=400, detail="Old password incorrect")
    await current_user.set_password(data.new_password)
    return {"status": "success"}

@router.put("/me/notifications")
async def update_notifications(data: NotificationSettingsRequest, current_user: User = Depends(get_current_active_user)):
    await current_user.update_profile(notifications=data.dict(exclude_unset=True))
    return {"status": "success"}

@router.put("/me/language")
async def update_language(data: LanguageSettingsRequest, current_user: User = Depends(get_current_active_user)):
    await current_user.update_profile(language=data.language, timezone=data.timezone)
    return {"status": "success"}

@router.get("/me/data")
async def download_user_data(current_user: User = Depends(get_current_active_user)):
    user_data = {
        "profile": current_user.to_dict(),
        # "history": await PredictionHistoryModel.find_by_user_id(str(current_user.id)),
    }
    return Response(
        content=json.dumps(user_data, default=str),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=user_data.json"}
    )

@router.delete("/me")
async def delete_account(current_user: User = Depends(get_current_active_user)):
    await current_user.delete()
    return {"status": "account deleted"} 