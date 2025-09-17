from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def system_status():
    return {"status": "active", "service": "system"}
