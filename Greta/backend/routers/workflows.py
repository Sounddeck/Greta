from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def workflows_status():
    return {"status": "active", "service": "workflows"}
