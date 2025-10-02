from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def multimodal_status():
    return {"status": "active", "service": "multimodal"}
