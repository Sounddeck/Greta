from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def agent_status():
    return {"status": "active", "service": "agent"}
