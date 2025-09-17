from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def memory_status():
    return {"status": "active", "service": "memory"}
