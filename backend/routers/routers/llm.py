from fastapi import APIRouter
router = APIRouter()
@router.get("/status")
async def llm_status():
    return {"status": "active", "service": "llm"}
