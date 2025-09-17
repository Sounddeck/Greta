from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def reasoning_status():
    return {"status": "active", "service": "reasoning"}
