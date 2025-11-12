"""
Advanced Analytics & Reporting Router for CPAS4
REST API endpoints for Analytics & Reporting
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from services.analytics_service import AnalyticsService
from models.analytics_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

async def get_service() -> AnalyticsService:
    db = Database()
    await db.connect()
    return AnalyticsService(db)

# Dashboard endpoints
@router.post("/dashboards", response_model=Dashboard, status_code=201)
async def create_dashboard(dashboard: DashboardCreate, svc: AnalyticsService = Depends(get_service)):
    return await svc.create_dashboard(dashboard)

@router.get("/dashboards/{dashboard_id}", response_model=Dashboard)
async def get_dashboard(dashboard_id: str, svc: AnalyticsService = Depends(get_service)):
    dash = await svc.get_dashboard(dashboard_id)
    if not dash: raise HTTPException(404, "Dashboard not found")
    return dash

@router.patch("/dashboards/{dashboard_id}", response_model=Dashboard)
async def update_dashboard(dashboard_id: str, dashboard_update: DashboardUpdate, svc: AnalyticsService = Depends(get_service)):
    dash = await svc.update_dashboard(dashboard_id, dashboard_update)
    if not dash: raise HTTPException(404, "Dashboard not found")
    return dash

@router.delete("/dashboards/{dashboard_id}", status_code=204)
async def delete_dashboard(dashboard_id: str, svc: AnalyticsService = Depends(get_service)):
    if not await svc.delete_dashboard(dashboard_id): raise HTTPException(404, "Dashboard not found")

@router.get("/dashboards", response_model=List[Dashboard])
async def list_dashboards(owner_id: Optional[str] = None, is_public: Optional[bool] = None, limit: int = Query(100, le=1000), svc: AnalyticsService = Depends(get_service)):
    return await svc.list_dashboards(owner_id, is_public, limit)

# Widget endpoints
@router.post("/dashboards/{dashboard_id}/widgets", response_model=DashboardWidget, status_code=201)
async def create_widget(dashboard_id: str, widget: DashboardWidgetCreate, svc: AnalyticsService = Depends(get_service)):
    # Verify dashboard exists
    dash = await svc.get_dashboard(dashboard_id)
    if not dash: raise HTTPException(404, "Dashboard not found")
    return await svc.create_widget(widget, dashboard_id)

@router.get("/widgets/{widget_id}", response_model=DashboardWidget)
async def get_widget(widget_id: str, svc: AnalyticsService = Depends(get_service)):
    widget = await svc.get_widget(widget_id)
    if not widget: raise HTTPException(404, "Widget not found")
    return widget

@router.patch("/widgets/{widget_id}", response_model=DashboardWidget)
async def update_widget(widget_id: str, widget_update: DashboardWidgetUpdate, svc: AnalyticsService = Depends(get_service)):
    widget = await svc.update_widget(widget_id, widget_update)
    if not widget: raise HTTPException(404, "Widget not found")
    return widget

@router.delete("/widgets/{widget_id}", status_code=204)
async def delete_widget(widget_id: str, svc: AnalyticsService = Depends(get_service)):
    if not await svc.delete_widget(widget_id): raise HTTPException(404, "Widget not found")

@router.get("/widgets", response_model=List[DashboardWidget])
async def list_widgets(dashboard_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: AnalyticsService = Depends(get_service)):
    return await svc.list_widgets(dashboard_id, limit)

# Report endpoints
@router.post("/reports", response_model=Report, status_code=201)
async def create_report(report: ReportCreate, svc: AnalyticsService = Depends(get_service)):
    return await svc.create_report(report)

@router.get("/reports/{report_id}", response_model=Report)
async def get_report(report_id: str, svc: AnalyticsService = Depends(get_service)):
    rpt = await svc.get_report(report_id)
    if not rpt: raise HTTPException(404, "Report not found")
    return rpt

@router.patch("/reports/{report_id}", response_model=Report)
async def update_report(report_id: str, report_update: ReportUpdate, svc: AnalyticsService = Depends(get_service)):
    rpt = await svc.update_report(report_id, report_update)
    if not rpt: raise HTTPException(404, "Report not found")
    return rpt

@router.delete("/reports/{report_id}", status_code=204)
async def delete_report(report_id: str, svc: AnalyticsService = Depends(get_service)):
    if not await svc.delete_report(report_id): raise HTTPException(404, "Report not found")

@router.get("/reports", response_model=List[Report])
async def list_reports(report_type: Optional[ReportType] = None, limit: int = Query(100, le=1000), svc: AnalyticsService = Depends(get_service)):
    return await svc.list_reports(report_type, limit)

# Report Execution endpoints
@router.post("/reports/{report_id}/execute", response_model=ReportExecution, status_code=201)
async def execute_report(report_id: str, parameters: Dict[str, Any] = None, svc: AnalyticsService = Depends(get_service)):
    try:
        return await svc.execute_report(report_id, parameters or {})
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get("/executions", response_model=List[ReportExecution])
async def list_executions(report_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: AnalyticsService = Depends(get_service)):
    return await svc.list_executions(report_id, limit)

# KPI endpoints
@router.post("/kpis", response_model=KPI, status_code=201)
async def create_kpi(kpi: KPICreate, svc: AnalyticsService = Depends(get_service)):
    return await svc.create_kpi(kpi)

@router.get("/kpis/{kpi_id}", response_model=KPI)
async def get_kpi(kpi_id: str, svc: AnalyticsService = Depends(get_service)):
    kpi = await svc.get_kpi(kpi_id)
    if not kpi: raise HTTPException(404, "KPI not found")
    return kpi

@router.patch("/kpis/{kpi_id}", response_model=KPI)
async def update_kpi(kpi_id: str, kpi_update: KPIUpdate, svc: AnalyticsService = Depends(get_service)):
    kpi = await svc.update_kpi(kpi_id, kpi_update)
    if not kpi: raise HTTPException(404, "KPI not found")
    return kpi

@router.delete("/kpis/{kpi_id}", status_code=204)
async def delete_kpi(kpi_id: str, svc: AnalyticsService = Depends(get_service)):
    if not await svc.delete_kpi(kpi_id): raise HTTPException(404, "KPI not found")

@router.get("/kpis", response_model=List[KPI])
async def list_kpis(data_source: Optional[str] = None, status: Optional[str] = None, limit: int = Query(100, le=1000), svc: AnalyticsService = Depends(get_service)):
    return await svc.list_kpis(data_source, status, limit)

# Cross-Module Analytics
@router.get("/cross-module", response_model=CrossModuleAnalytics)
async def get_cross_module_analytics(
    start_date: Optional[str] = Query(None, description="ISO format date"),
    end_date: Optional[str] = Query(None, description="ISO format date"),
    svc: AnalyticsService = Depends(get_service)
):
    start = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
    end = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()
    return await svc.get_cross_module_analytics(start, end)

# Statistics
@router.get("/stats", response_model=AnalyticsStats)
async def get_stats(svc: AnalyticsService = Depends(get_service)):
    return await svc.get_stats()

# Health check
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "module": "Analytics",
        "features": [
            "Dashboards",
            "Widgets",
            "Reports",
            "Report Execution",
            "KPIs",
            "Cross-Module Analytics",
            "Statistics"
        ],
        "endpoints": 28,
        "database": "MongoDB/File Storage"
    }

