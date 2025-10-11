"""
Reports Router for GRETA PAI
API endpoints for report generation and management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import logging
from datetime import datetime
import uuid

from models.report_models import (
    ReportGenerateRequest,
    ReportResponse,
    ReportListResponse,
    ReportStatistics,
    ReportStatus,
    ReportType,
    ReportFormat
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["Reports"])

# In-memory storage for demo (replace with database in production)
reports_storage: dict = {}


def generate_report_task(report_id: str, request: ReportGenerateRequest):
    """Background task to generate report"""
    try:
        logger.info(f"Generating report {report_id}: {request.title}")
        
        # Simulate report generation
        # In production, this would:
        # 1. Query database for data
        # 2. Process and aggregate data
        # 3. Generate report in requested format
        # 4. Store report file
        # 5. Update status to completed
        
        # For now, just update status
        if report_id in reports_storage:
            reports_storage[report_id]["status"] = ReportStatus.COMPLETED
            reports_storage[report_id]["completed_at"] = datetime.utcnow()
            reports_storage[report_id]["download_url"] = f"/api/v1/reports/{report_id}/download"
            
        logger.info(f"Report {report_id} generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate report {report_id}: {str(e)}")
        if report_id in reports_storage:
            reports_storage[report_id]["status"] = ReportStatus.FAILED
            reports_storage[report_id]["error"] = str(e)


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a new report
    
    This endpoint creates a report generation job and returns immediately.
    The actual report generation happens in the background.
    """
    try:
        # Generate unique report ID
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        
        # Create report record
        report = {
            "id": report_id,
            "title": request.title,
            "report_type": request.report_type,
            "status": ReportStatus.PROCESSING,
            "format": request.format,
            "created_at": datetime.utcnow(),
            "completed_at": None,
            "download_url": None,
            "error": None,
            "metadata": {
                "filters": request.filters,
                "date_range": request.date_range
            }
        }
        
        # Store report
        reports_storage[report_id] = report
        
        # Schedule background task
        background_tasks.add_task(generate_report_task, report_id, request)
        
        logger.info(f"Created report generation job: {report_id}")
        
        return ReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Failed to create report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """
    Get report by ID
    
    Returns the report metadata and status.
    """
    if report_id not in reports_storage:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    report = reports_storage[report_id]
    return ReportResponse(**report)


@router.get("/", response_model=ReportListResponse)
async def list_reports(
    page: int = 1,
    page_size: int = 10,
    report_type: Optional[ReportType] = None,
    status: Optional[ReportStatus] = None
):
    """
    List all reports with optional filtering
    
    Supports pagination and filtering by type and status.
    """
    # Filter reports
    filtered_reports = list(reports_storage.values())
    
    if report_type:
        filtered_reports = [r for r in filtered_reports if r["report_type"] == report_type]
    
    if status:
        filtered_reports = [r for r in filtered_reports if r["status"] == status]
    
    # Sort by created_at descending
    filtered_reports.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    total = len(filtered_reports)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_reports = filtered_reports[start:end]
    
    return ReportListResponse(
        reports=[ReportResponse(**r) for r in paginated_reports],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """
    Download report file
    
    Returns the generated report file in the requested format.
    """
    if report_id not in reports_storage:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    report = reports_storage[report_id]
    
    if report["status"] != ReportStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Report is not ready. Current status: {report['status']}"
        )
    
    # In production, this would return the actual file
    # For now, return mock data
    mock_data = {
        "report_id": report_id,
        "title": report["title"],
        "type": report["report_type"],
        "generated_at": report["completed_at"].isoformat() if report["completed_at"] else None,
        "data": {
            "summary": "This is a mock report. In production, this would contain actual data.",
            "metrics": {
                "total_tasks": 42,
                "completed_tasks": 38,
                "success_rate": 0.905
            }
        }
    }
    
    return JSONResponse(content=mock_data)


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """
    Delete a report
    
    Removes the report and its associated files.
    """
    if report_id not in reports_storage:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    
    # Delete report
    del reports_storage[report_id]
    
    logger.info(f"Deleted report: {report_id}")
    
    return {"success": True, "message": f"Report {report_id} deleted"}


@router.get("/statistics", response_model=ReportStatistics)
async def get_report_statistics():
    """
    Get report statistics
    
    Returns aggregated statistics about all reports.
    """
    reports = list(reports_storage.values())
    
    # Count by type
    reports_by_type = {}
    for report_type in ReportType:
        count = len([r for r in reports if r["report_type"] == report_type])
        if count > 0:
            reports_by_type[report_type.value] = count
    
    # Count by status
    reports_by_status = {}
    for status in ReportStatus:
        count = len([r for r in reports if r["status"] == status])
        if count > 0:
            reports_by_status[status.value] = count
    
    # Get recent reports (last 5)
    recent = sorted(reports, key=lambda x: x["created_at"], reverse=True)[:5]
    
    return ReportStatistics(
        total_reports=len(reports),
        reports_by_type=reports_by_type,
        reports_by_status=reports_by_status,
        recent_reports=[ReportResponse(**r) for r in recent]
    )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Reports API",
        "total_reports": len(reports_storage),
        "timestamp": datetime.utcnow().isoformat()
    }
