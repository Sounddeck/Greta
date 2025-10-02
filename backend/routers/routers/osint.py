from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def osint_status():
    return {"status": "active", "service": "osint"}
