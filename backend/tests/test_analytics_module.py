"""
Advanced Analytics & Reporting Module Test Suite
Comprehensive tests for Analytics & Reporting functionality
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from database import Database
from services.analytics_service import AnalyticsService
from models.analytics_models import *

class TestAnalyticsModule:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
    async def setup(self):
        """Initialize database and service"""
        self.db = Database()
        await self.db.connect()
        self.service = AnalyticsService(self.db)
        print("✅ Analytics module test setup complete")
    
    async def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            for collection in ["analytics_dashboards", "analytics_widgets", "analytics_reports", "analytics_executions", "analytics_kpis"]:
                try:
                    docs = await self.db.find_many(collection, {}, limit=10000)
                    for doc in docs:
                        if "_id" in doc:
                            await self.db.delete_one(collection, {"_id": doc["_id"]})
                except:
                    pass
        print("✅ Test cleanup complete")
    
    def record_result(self, test_name: str, passed: bool, message: str = ""):
        """Record test result"""
        self.test_results.append({"test": test_name, "passed": passed, "message": message})
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}" + (f" - {message}" if message else ""))
    
    async def test_dashboard_crud(self):
        """Test dashboard CRUD operations"""
        try:
            # Create dashboard
            dashboard_create = DashboardCreate(
                name="Test Dashboard",
                description="A test dashboard",
                is_default=True,
                is_public=True,
                owner_id="user_001",
                tags=["test", "analytics"]
            )
            dashboard = await self.service.create_dashboard(dashboard_create)
            assert dashboard.dashboard_id.startswith("dash_"), "Dashboard ID format incorrect"
            assert dashboard.name == "Test Dashboard", "Dashboard name mismatch"
            
            # Get dashboard
            retrieved = await self.service.get_dashboard(dashboard.dashboard_id)
            assert retrieved is not None, "Dashboard not found"
            assert retrieved.dashboard_id == dashboard.dashboard_id, "Dashboard ID mismatch"
            
            # Update dashboard
            update = DashboardUpdate(name="Updated Dashboard", is_public=False)
            updated = await self.service.update_dashboard(dashboard.dashboard_id, update)
            assert updated is not None, "Dashboard update failed"
            assert updated.name == "Updated Dashboard", "Dashboard name not updated"
            
            # List dashboards
            dashboards = await self.service.list_dashboards()
            assert len(dashboards) >= 1, "No dashboards found"
            
            # Delete dashboard
            deleted = await self.service.delete_dashboard(dashboard.dashboard_id)
            assert deleted, "Dashboard deletion failed"
            
            self.record_result("Dashboard CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Dashboard CRUD", False, str(e))
    
    async def test_widget_crud(self):
        """Test widget CRUD operations"""
        try:
            # Create dashboard first
            dashboard_create = DashboardCreate(name="Widget Test Dashboard", owner_id="user_001")
            dashboard = await self.service.create_dashboard(dashboard_create)
            
            # Create widget
            widget_create = DashboardWidgetCreate(
                title="Sales Chart",
                widget_type=ChartType.LINE,
                data_source="crm",
                metric="total_deals",
                aggregation=AggregationType.SUM,
                time_period=TimePeriod.THIS_MONTH
            )
            widget = await self.service.create_widget(widget_create, dashboard.dashboard_id)
            assert widget.widget_id.startswith("wdg_"), "Widget ID format incorrect"
            assert widget.title == "Sales Chart", "Widget title mismatch"
            
            # Get widget
            retrieved = await self.service.get_widget(widget.widget_id)
            assert retrieved is not None, "Widget not found"
            
            # Update widget
            update = DashboardWidgetUpdate(title="Updated Sales Chart", refresh_interval=600)
            updated = await self.service.update_widget(widget.widget_id, update)
            assert updated is not None, "Widget update failed"
            assert updated.title == "Updated Sales Chart", "Widget title not updated"
            
            # List widgets
            widgets = await self.service.list_widgets(dashboard_id=dashboard.dashboard_id)
            assert len(widgets) >= 1, "No widgets found"
            
            self.record_result("Widget CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Widget CRUD", False, str(e))
    
    async def test_report_crud(self):
        """Test report CRUD operations"""
        try:
            # Create report
            report_create = ReportCreate(
                name="Monthly Sales Report",
                report_type=ReportType.SALES,
                description="Monthly sales analysis",
                data_sources=["crm", "finance"],
                metrics=["total_revenue", "total_deals"],
                dimensions=["month", "region"],
                time_period=TimePeriod.THIS_MONTH,
                format=ReportFormat.JSON
            )
            report = await self.service.create_report(report_create)
            assert report.report_id.startswith("rpt_"), "Report ID format incorrect"
            assert report.name == "Monthly Sales Report", "Report name mismatch"
            
            # Get report
            retrieved = await self.service.get_report(report.report_id)
            assert retrieved is not None, "Report not found"
            
            # Update report
            update = ReportUpdate(name="Updated Sales Report", format=ReportFormat.CSV)
            updated = await self.service.update_report(report.report_id, update)
            assert updated is not None, "Report update failed"
            assert updated.name == "Updated Sales Report", "Report name not updated"
            
            # List reports
            reports = await self.service.list_reports()
            assert len(reports) >= 1, "No reports found"
            
            self.record_result("Report CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Report CRUD", False, str(e))
    
    async def test_report_execution(self):
        """Test report execution"""
        try:
            # Create report
            report_create = ReportCreate(
                name="Test Execution Report",
                report_type=ReportType.FINANCIAL,
                data_sources=["finance"],
                metrics=["total_revenue"],
                format=ReportFormat.JSON
            )
            report = await self.service.create_report(report_create)
            
            # Execute report
            execution = await self.service.execute_report(report.report_id, {"test_param": "value"})
            assert execution.execution_id.startswith("exec_"), "Execution ID format incorrect"
            assert execution.status == "completed", "Execution status incorrect"
            assert execution.duration_seconds is not None, "Duration not calculated"
            
            # List executions
            executions = await self.service.list_executions(report_id=report.report_id)
            assert len(executions) >= 1, "No executions found"
            
            # Verify report stats updated
            updated_report = await self.service.get_report(report.report_id)
            assert updated_report.run_count == 1, "Run count not updated"
            assert updated_report.last_run is not None, "Last run not updated"
            
            self.record_result("Report Execution", True, "Execution successful")
        except Exception as e:
            self.record_result("Report Execution", False, str(e))
    
    async def test_kpi_crud(self):
        """Test KPI CRUD operations"""
        try:
            # Create KPI
            kpi_create = KPICreate(
                name="Monthly Revenue Target",
                description="Monthly revenue goal",
                data_source="finance",
                metric="total_revenue",
                target_value=100000.0,
                current_value=75000.0,
                unit="USD",
                threshold_warning=80.0,
                threshold_critical=60.0,
                is_higher_better=True
            )
            kpi = await self.service.create_kpi(kpi_create)
            assert kpi.kpi_id.startswith("kpi_"), "KPI ID format incorrect"
            assert kpi.name == "Monthly Revenue Target", "KPI name mismatch"
            assert kpi.achievement_percent == 75.0, "Achievement calculation incorrect"
            assert kpi.status == "warning", "KPI status incorrect"
            
            # Get KPI
            retrieved = await self.service.get_kpi(kpi.kpi_id)
            assert retrieved is not None, "KPI not found"
            
            # Update KPI
            update = KPIUpdate(current_value=90000.0)
            updated = await self.service.update_kpi(kpi.kpi_id, update)
            assert updated is not None, "KPI update failed"
            assert updated.achievement_percent == 90.0, "Achievement not recalculated"
            assert updated.status == "on_track", "Status not updated"
            
            # List KPIs
            kpis = await self.service.list_kpis()
            assert len(kpis) >= 1, "No KPIs found"
            
            self.record_result("KPI CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("KPI CRUD", False, str(e))
    
    async def test_cross_module_analytics(self):
        """Test cross-module analytics"""
        try:
            start_date = datetime.utcnow() - timedelta(days=30)
            end_date = datetime.utcnow()
            
            analytics = await self.service.get_cross_module_analytics(start_date, end_date)
            assert analytics is not None, "Analytics not returned"
            assert analytics.period_start == start_date, "Start date mismatch"
            assert analytics.period_end == end_date, "End date mismatch"
            assert hasattr(analytics, 'total_contacts'), "Missing CRM metrics"
            assert hasattr(analytics, 'total_revenue'), "Missing finance metrics"
            assert hasattr(analytics, 'total_projects'), "Missing project metrics"
            
            self.record_result("Cross-Module Analytics", True, "Analytics generated successfully")
        except Exception as e:
            self.record_result("Cross-Module Analytics", False, str(e))
    
    async def test_statistics(self):
        """Test analytics statistics"""
        try:
            stats = await self.service.get_stats()
            assert stats is not None, "Stats not returned"
            assert hasattr(stats, 'total_dashboards'), "Missing total_dashboards"
            assert hasattr(stats, 'total_reports'), "Missing total_reports"
            assert hasattr(stats, 'total_widgets'), "Missing total_widgets"
            assert hasattr(stats, 'total_kpis'), "Missing total_kpis"
            
            self.record_result("Statistics", True, f"Stats: {stats.total_dashboards} dashboards, {stats.total_reports} reports, {stats.total_kpis} KPIs")
        except Exception as e:
            self.record_result("Statistics", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 ADVANCED ANALYTICS & REPORTING MODULE TEST SUITE")
        print("="*80 + "\n")
        
        await self.setup()
        
        # Run tests
        await self.test_dashboard_crud()
        await self.test_widget_crud()
        await self.test_report_crud()
        await self.test_report_execution()
        await self.test_kpi_crud()
        await self.test_cross_module_analytics()
        await self.test_statistics()
        
        await self.cleanup()
        
        # Print summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass Rate: {(passed/total*100):.1f}%")
        print("="*80 + "\n")
        
        return passed == total

async def main():
    tester = TestAnalyticsModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

