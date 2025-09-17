from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def voice_status():
    return {"status": "active", "service": "voice"}
