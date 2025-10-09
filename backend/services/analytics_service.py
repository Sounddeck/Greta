"""
Advanced Analytics & Reporting Service for CPAS4 - Real Database Integration
Cross-module analytics, dashboards, and reporting capabilities
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from database import Database
from models.analytics_models import *

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self, database: Database):
        self.db = database
        self.dashboards_collection = "analytics_dashboards"
        self.widgets_collection = "analytics_widgets"
        self.reports_collection = "analytics_reports"
        self.executions_collection = "analytics_executions"
        self.kpis_collection = "analytics_kpis"
        logger.info("✅ Analytics Service initialized with real database")
    
    # Dashboard CRUD
    async def create_dashboard(self, dashboard_create: DashboardCreate) -> Dashboard:
        dashboard_id = f"dash_{uuid.uuid4().hex[:12]}"
        dashboard = Dashboard(dashboard_id=dashboard_id, **dashboard_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        dd = dashboard.model_dump()
        for k in ["created_at", "updated_at"]:
            if dd.get(k): dd[k] = dd[k].isoformat()
        await self.db.insert_one(self.dashboards_collection, dd)
        logger.info(f"✅ Dashboard created: {dashboard_id}")
        return dashboard
    
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        dd = await self.db.find_one(self.dashboards_collection, {"dashboard_id": dashboard_id})
        if not dd: return None
        for k in ["created_at", "updated_at"]:
            if k in dd and dd[k] and isinstance(dd[k], str): dd[k] = datetime.fromisoformat(dd[k])
        return Dashboard(**dd)
    
    async def update_dashboard(self, dashboard_id: str, dashboard_update: DashboardUpdate) -> Optional[Dashboard]:
        ud = dashboard_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_dashboard(dashboard_id) if await self.db.update_one(self.dashboards_collection, {"dashboard_id": dashboard_id}, {"$set": ud}) else None
    
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        # Delete associated widgets
        widgets = await self.list_widgets(dashboard_id=dashboard_id, limit=10000)
        for widget in widgets:
            await self.delete_widget(widget.widget_id)
        return await self.db.delete_one(self.dashboards_collection, {"dashboard_id": dashboard_id})
    
    async def list_dashboards(self, owner_id: Optional[str] = None, is_public: Optional[bool] = None, limit: int = 100) -> List[Dashboard]:
        f = {}
        if owner_id: f["owner_id"] = owner_id
        if is_public is not None: f["is_public"] = is_public
        dds = await self.db.find_many(self.dashboards_collection, f, limit=limit)
        dashboards = []
        for dd in dds:
            for k in ["created_at", "updated_at"]:
                if k in dd and dd[k] and isinstance(dd[k], str): dd[k] = datetime.fromisoformat(dd[k])
            # Count widgets
            widgets = await self.list_widgets(dashboard_id=dd["dashboard_id"], limit=10000)
            dd["widget_count"] = len(widgets)
            dashboards.append(Dashboard(**dd))
        return dashboards
    
    # Widget CRUD
    async def create_widget(self, widget_create: DashboardWidgetCreate, dashboard_id: str) -> DashboardWidget:
        widget_id = f"wdg_{uuid.uuid4().hex[:12]}"
        widget = DashboardWidget(widget_id=widget_id, dashboard_id=dashboard_id, **widget_create.model_dump(), last_updated=datetime.utcnow(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        wd = widget.model_dump()
        for k in ["created_at", "updated_at", "last_updated", "custom_start_date", "custom_end_date"]:
            if wd.get(k): wd[k] = wd[k].isoformat()
        await self.db.insert_one(self.widgets_collection, wd)
        
        # Update dashboard widget count
        dashboard = await self.get_dashboard(dashboard_id)
        if dashboard:
            await self.db.update_one(self.dashboards_collection, {"dashboard_id": dashboard_id}, {"$set": {"widget_count": dashboard.widget_count + 1}})
        
        logger.info(f"✅ Widget created: {widget_id}")
        return widget
    
    async def get_widget(self, widget_id: str) -> Optional[DashboardWidget]:
        wd = await self.db.find_one(self.widgets_collection, {"widget_id": widget_id})
        if not wd: return None
        for k in ["created_at", "updated_at", "last_updated", "custom_start_date", "custom_end_date"]:
            if k in wd and wd[k] and isinstance(wd[k], str): wd[k] = datetime.fromisoformat(wd[k])
        return DashboardWidget(**wd)
    
    async def update_widget(self, widget_id: str, widget_update: DashboardWidgetUpdate) -> Optional[DashboardWidget]:
        ud = widget_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "custom_start_date" in ud and ud["custom_start_date"]: ud["custom_start_date"] = ud["custom_start_date"].isoformat()
        if "custom_end_date" in ud and ud["custom_end_date"]: ud["custom_end_date"] = ud["custom_end_date"].isoformat()
        return await self.get_widget(widget_id) if await self.db.update_one(self.widgets_collection, {"widget_id": widget_id}, {"$set": ud}) else None
    
    async def delete_widget(self, widget_id: str) -> bool:
        widget = await self.get_widget(widget_id)
        if widget and await self.db.delete_one(self.widgets_collection, {"widget_id": widget_id}):
            # Update dashboard widget count
            dashboard = await self.get_dashboard(widget.dashboard_id)
            if dashboard:
                await self.db.update_one(self.dashboards_collection, {"dashboard_id": widget.dashboard_id}, {"$set": {"widget_count": max(0, dashboard.widget_count - 1)}})
            return True
        return False
    
    async def list_widgets(self, dashboard_id: Optional[str] = None, limit: int = 100) -> List[DashboardWidget]:
        f = {}
        if dashboard_id: f["dashboard_id"] = dashboard_id
        wds = await self.db.find_many(self.widgets_collection, f, limit=limit)
        widgets = []
        for wd in wds:
            for k in ["created_at", "updated_at", "last_updated", "custom_start_date", "custom_end_date"]:
                if k in wd and wd[k] and isinstance(wd[k], str): wd[k] = datetime.fromisoformat(wd[k])
            widgets.append(DashboardWidget(**wd))
        return widgets
    
    # Report CRUD
    async def create_report(self, report_create: ReportCreate) -> Report:
        report_id = f"rpt_{uuid.uuid4().hex[:12]}"
        report = Report(report_id=report_id, **report_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        rd = report.model_dump()
        for k in ["created_at", "updated_at", "last_run", "custom_start_date", "custom_end_date"]:
            if rd.get(k): rd[k] = rd[k].isoformat()
        await self.db.insert_one(self.reports_collection, rd)
        logger.info(f"✅ Report created: {report_id}")
        return report
    
    async def get_report(self, report_id: str) -> Optional[Report]:
        rd = await self.db.find_one(self.reports_collection, {"report_id": report_id})
        if not rd: return None
        for k in ["created_at", "updated_at", "last_run", "custom_start_date", "custom_end_date"]:
            if k in rd and rd[k] and isinstance(rd[k], str): rd[k] = datetime.fromisoformat(rd[k])
        return Report(**rd)
    
    async def update_report(self, report_id: str, report_update: ReportUpdate) -> Optional[Report]:
        ud = report_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "custom_start_date" in ud and ud["custom_start_date"]: ud["custom_start_date"] = ud["custom_start_date"].isoformat()
        if "custom_end_date" in ud and ud["custom_end_date"]: ud["custom_end_date"] = ud["custom_end_date"].isoformat()
        return await self.get_report(report_id) if await self.db.update_one(self.reports_collection, {"report_id": report_id}, {"$set": ud}) else None
    
    async def delete_report(self, report_id: str) -> bool:
        return await self.db.delete_one(self.reports_collection, {"report_id": report_id})
    
    async def list_reports(self, report_type: Optional[ReportType] = None, limit: int = 100) -> List[Report]:
        f = {}
        if report_type: f["report_type"] = report_type
        rds = await self.db.find_many(self.reports_collection, f, limit=limit)
        reports = []
        for rd in rds:
            for k in ["created_at", "updated_at", "last_run", "custom_start_date", "custom_end_date"]:
                if k in rd and rd[k] and isinstance(rd[k], str): rd[k] = datetime.fromisoformat(rd[k])
            reports.append(Report(**rd))
        return reports
    
    # Report Execution
    async def execute_report(self, report_id: str, parameters: Dict[str, Any] = None) -> ReportExecution:
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        report = await self.get_report(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        execution = ReportExecution(
            execution_id=execution_id,
            report_id=report_id,
            parameters=parameters or {},
            started_at=datetime.utcnow(),
            status="running"
        )
        
        # Simulate report execution (in real implementation, this would fetch and process data)
        execution.data = {"message": "Report execution simulated", "report_name": report.name, "report_type": report.report_type}
        execution.row_count = 0
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        execution.status = "completed"
        
        ed = execution.model_dump()
        for k in ["started_at", "completed_at"]:
            if ed.get(k): ed[k] = ed[k].isoformat()
        await self.db.insert_one(self.executions_collection, ed)
        
        # Update report stats
        await self.db.update_one(self.reports_collection, {"report_id": report_id}, {"$set": {"last_run": datetime.utcnow().isoformat(), "run_count": report.run_count + 1}})
        
        logger.info(f"✅ Report executed: {execution_id}")
        return execution
    
    async def list_executions(self, report_id: Optional[str] = None, limit: int = 100) -> List[ReportExecution]:
        f = {}
        if report_id: f["report_id"] = report_id
        eds = await self.db.find_many(self.executions_collection, f, limit=limit)
        executions = []
        for ed in eds:
            for k in ["started_at", "completed_at"]:
                if k in ed and ed[k] and isinstance(ed[k], str): ed[k] = datetime.fromisoformat(ed[k])
            executions.append(ReportExecution(**ed))
        return executions
    
    # KPI CRUD
    async def create_kpi(self, kpi_create: KPICreate) -> KPI:
        kpi_id = f"kpi_{uuid.uuid4().hex[:12]}"
        achievement = (kpi_create.current_value / kpi_create.target_value * 100) if kpi_create.target_value > 0 else 0
        status = "on_track"
        if kpi_create.threshold_critical and achievement < kpi_create.threshold_critical:
            status = "critical"
        elif kpi_create.threshold_warning and achievement < kpi_create.threshold_warning:
            status = "warning"
        
        kpi = KPI(kpi_id=kpi_id, **kpi_create.model_dump(), achievement_percent=achievement, status=status, last_updated=datetime.utcnow(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        kd = kpi.model_dump()
        for k in ["created_at", "updated_at", "last_updated"]:
            if kd.get(k): kd[k] = kd[k].isoformat()
        await self.db.insert_one(self.kpis_collection, kd)
        logger.info(f"✅ KPI created: {kpi_id}")
        return kpi
    
    async def get_kpi(self, kpi_id: str) -> Optional[KPI]:
        kd = await self.db.find_one(self.kpis_collection, {"kpi_id": kpi_id})
        if not kd: return None
        for k in ["created_at", "updated_at", "last_updated"]:
            if k in kd and kd[k] and isinstance(kd[k], str): kd[k] = datetime.fromisoformat(kd[k])
        return KPI(**kd)
    
    async def update_kpi(self, kpi_id: str, kpi_update: KPIUpdate) -> Optional[KPI]:
        kpi = await self.get_kpi(kpi_id)
        if not kpi: return None
        
        ud = kpi_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        ud["last_updated"] = datetime.utcnow().isoformat()
        
        # Recalculate achievement
        target = ud.get("target_value", kpi.target_value)
        current = ud.get("current_value", kpi.current_value)
        achievement = (current / target * 100) if target > 0 else 0
        ud["achievement_percent"] = achievement
        
        # Update status
        threshold_critical = ud.get("threshold_critical", kpi.threshold_critical)
        threshold_warning = ud.get("threshold_warning", kpi.threshold_warning)
        status = "on_track"
        if threshold_critical and achievement < threshold_critical:
            status = "critical"
        elif threshold_warning and achievement < threshold_warning:
            status = "warning"
        ud["status"] = status
        
        return await self.get_kpi(kpi_id) if await self.db.update_one(self.kpis_collection, {"kpi_id": kpi_id}, {"$set": ud}) else None
    
    async def delete_kpi(self, kpi_id: str) -> bool:
        return await self.db.delete_one(self.kpis_collection, {"kpi_id": kpi_id})
    
    async def list_kpis(self, data_source: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[KPI]:
        f = {}
        if data_source: f["data_source"] = data_source
        if status: f["status"] = status
        kds = await self.db.find_many(self.kpis_collection, f, limit=limit)
        kpis = []
        for kd in kds:
            for k in ["created_at", "updated_at", "last_updated"]:
                if k in kd and kd[k] and isinstance(kd[k], str): kd[k] = datetime.fromisoformat(kd[k])
            kpis.append(KPI(**kd))
        return kpis
    
    # Cross-Module Analytics
    async def get_cross_module_analytics(self, start_date: datetime, end_date: datetime) -> CrossModuleAnalytics:
        # This would aggregate data from all modules
        # For now, return simulated data
        analytics = CrossModuleAnalytics(
            period_start=start_date,
            period_end=end_date,
            total_contacts=0,
            total_companies=0,
            total_deals=0,
            deals_value=0.0,
            total_candidates=0,
            total_jobs=0,
            total_applications=0,
            total_revenue=0.0,
            total_expenses=0.0,
            net_profit=0.0,
            total_projects=0,
            active_projects=0,
            completed_projects=0,
            total_project_hours=0.0,
            total_products=0,
            total_stock_value=0.0,
            low_stock_alerts=0,
            total_agents=0,
            total_conversations=0,
            avg_response_time=0.0
        )
        return analytics
    
    # Statistics
    async def get_stats(self) -> AnalyticsStats:
        all_dashboards = await self.list_dashboards(limit=10000)
        all_reports = await self.list_reports(limit=10000)
        all_widgets = await self.list_widgets(limit=10000)
        all_kpis = await self.list_kpis(limit=10000)
        all_executions = await self.list_executions(limit=10000)
        
        dashboards_by_owner = {}
        for d in all_dashboards:
            dashboards_by_owner[d.owner_id] = dashboards_by_owner.get(d.owner_id, 0) + 1
        
        reports_by_type = {t.value: sum(1 for r in all_reports if r.report_type == t) for t in ReportType}
        kpis_by_status = {"on_track": sum(1 for k in all_kpis if k.status == "on_track"), "warning": sum(1 for k in all_kpis if k.status == "warning"), "critical": sum(1 for k in all_kpis if k.status == "critical")}
        
        recent_executions = sum(1 for e in all_executions if e.started_at > datetime.utcnow() - timedelta(days=7))
        
        most_used = sorted(all_reports, key=lambda r: r.run_count, reverse=True)[:5]
        most_used_data = [{"report_id": r.report_id, "name": r.name, "run_count": r.run_count} for r in most_used]
        
        recent = sorted(all_reports, key=lambda r: r.created_at, reverse=True)[:5]
        recent_data = [{"report_id": r.report_id, "name": r.name, "created_at": r.created_at.isoformat()} for r in recent]
        
        return AnalyticsStats(
            total_dashboards=len(all_dashboards),
            total_reports=len(all_reports),
            total_widgets=len(all_widgets),
            total_kpis=len(all_kpis),
            total_report_executions=len(all_executions),
            recent_executions=recent_executions,
            dashboards_by_owner=dashboards_by_owner,
            reports_by_type=reports_by_type,
            kpis_by_status=kpis_by_status,
            most_used_reports=most_used_data,
            recent_reports=recent_data
        )

