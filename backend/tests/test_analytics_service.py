"""
Comprehensive tests for analytics service

Tests cover:
- Dashboard CRUD operations
- Widget management  
- Report creation and execution
- KPI tracking and metrics
- Error handling
"""

import pytest
from unittest.mock import AsyncMock
from datetime import datetime


class TestDashboardOperations:
    """Test dashboard CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_create_dashboard(self):
        """Test creating a dashboard"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import DashboardCreate
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)
        
        # Create analytics service
        analytics_service = AnalyticsService(mock_db)
        
        # Create dashboard
        dashboard_create = DashboardCreate(
            name="Test Dashboard",
            description="Test dashboard description",
            owner_id="user_123",
            is_public=False
        )
        
        dashboard = await analytics_service.create_dashboard(dashboard_create)
        
        assert dashboard is not None
        assert dashboard.name == "Test Dashboard"
        assert dashboard.owner_id == "user_123"
        assert dashboard.dashboard_id.startswith("dash_")
    
    @pytest.mark.asyncio
    async def test_get_dashboard(self):
        """Test getting a dashboard"""
        from services.analytics_service import AnalyticsService
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_one = AsyncMock(return_value={
            "dashboard_id": "dash_123",
            "name": "Test Dashboard",
            "description": "Test description",
            "owner_id": "user_123",
            "is_public": False,
            "widget_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Create analytics service
        analytics_service = AnalyticsService(mock_db)
        
        # Get dashboard
        dashboard = await analytics_service.get_dashboard("dash_123")
        
        assert dashboard is not None
        assert dashboard.dashboard_id == "dash_123"
        assert dashboard.name == "Test Dashboard"
    
    @pytest.mark.asyncio
    async def test_update_dashboard(self):
        """Test updating a dashboard"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import DashboardUpdate
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.update_one = AsyncMock(return_value=True)
        mock_db.find_one = AsyncMock(return_value={
            "dashboard_id": "dash_123",
            "name": "Updated Dashboard",
            "description": "Updated description",
            "owner_id": "user_123",
            "is_public": True,
            "widget_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Create analytics service
        analytics_service = AnalyticsService(mock_db)
        
        # Update dashboard
        dashboard_update = DashboardUpdate(
            name="Updated Dashboard",
            description="Updated description",
            is_public=True
        )
        
        dashboard = await analytics_service.update_dashboard("dash_123", dashboard_update)
        
        assert dashboard is not None
        assert dashboard.name == "Updated Dashboard"
        assert dashboard.is_public is True
    
    @pytest.mark.asyncio
    async def test_delete_dashboard(self):
        """Test deleting a dashboard"""
        from services.analytics_service import AnalyticsService
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.delete_one = AsyncMock(return_value=True)
        mock_db.find_many = AsyncMock(return_value=[])  # No widgets
        
        # Create analytics service
        analytics_service = AnalyticsService(mock_db)
        
        # Delete dashboard
        result = await analytics_service.delete_dashboard("dash_123")

        assert result is True


class TestReportOperations:
    """Test report creation and execution"""

    @pytest.mark.asyncio
    async def test_create_report(self):
        """Test creating a report"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import ReportCreate, ReportType, TimePeriod

        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Create report
        report_create = ReportCreate(
            name="Test Report",
            description="Test report description",
            report_type=ReportType.CRM,
            data_sources=["crm"],
            metrics=["total_contacts", "total_deals"],
            time_period=TimePeriod.THIS_MONTH
        )

        report = await analytics_service.create_report(report_create)

        assert report is not None
        assert report.name == "Test Report"
        assert report.report_id.startswith("rpt_")
        assert report.report_type == ReportType.CRM

    @pytest.mark.asyncio
    async def test_execute_report(self):
        """Test executing a report"""
        from services.analytics_service import AnalyticsService

        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_one = AsyncMock(return_value={
            "report_id": "rpt_123",
            "name": "Test Report",
            "description": "Test",
            "report_type": "crm",
            "data_sources": ["crm"],
            "metrics": ["total_contacts"],
            "dimensions": [],
            "filters": [],
            "time_period": "this_month",
            "format": "json",
            "recipients": [],
            "metadata": {},
            "run_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": "system"
        })
        mock_db.insert_one = AsyncMock(return_value=True)
        mock_db.update_one = AsyncMock(return_value=True)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Execute report
        execution = await analytics_service.execute_report("rpt_123")

        assert execution is not None
        assert execution.execution_id.startswith("exec_")
        assert execution.report_id == "rpt_123"
        assert execution.status == "completed"
        assert execution.data is not None

    @pytest.mark.asyncio
    async def test_delete_report(self):
        """Test deleting a report"""
        from services.analytics_service import AnalyticsService

        # Create mock database
        mock_db = AsyncMock()
        mock_db.delete_one = AsyncMock(return_value=True)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Delete report
        result = await analytics_service.delete_report("rpt_123")

        assert result is True


class TestKPITracking:
    """Test KPI tracking and metrics"""

    @pytest.mark.asyncio
    async def test_create_kpi(self):
        """Test creating a KPI"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import KPICreate

        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Create KPI
        kpi_create = KPICreate(
            name="Test KPI",
            description="Test KPI description",
            data_source="crm",
            metric="total_contacts",
            current_value=75.0,
            target_value=100.0,
            threshold_warning=80.0,
            threshold_critical=60.0
        )

        kpi = await analytics_service.create_kpi(kpi_create)

        assert kpi is not None
        assert kpi.name == "Test KPI"
        assert kpi.kpi_id.startswith("kpi_")
        assert kpi.achievement_percent == 75.0
        assert kpi.status == "warning"  # Below warning threshold

    @pytest.mark.asyncio
    async def test_kpi_status_calculation(self):
        """Test KPI status calculation"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import KPICreate

        # Create mock database
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(return_value=True)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Create KPI with critical status
        kpi_create = KPICreate(
            name="Critical KPI",
            description="Test",
            data_source="crm",
            metric="total_contacts",
            current_value=50.0,
            target_value=100.0,
            threshold_warning=80.0,
            threshold_critical=60.0
        )

        kpi = await analytics_service.create_kpi(kpi_create)

        assert kpi.status == "critical"  # Below critical threshold
        assert kpi.achievement_percent == 50.0


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_get_nonexistent_dashboard(self):
        """Test getting nonexistent dashboard"""
        from services.analytics_service import AnalyticsService

        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_one = AsyncMock(return_value=None)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Get nonexistent dashboard
        dashboard = await analytics_service.get_dashboard("nonexistent")

        assert dashboard is None

    @pytest.mark.asyncio
    async def test_execute_nonexistent_report(self):
        """Test executing nonexistent report"""
        from services.analytics_service import AnalyticsService

        # Create mock database
        mock_db = AsyncMock()
        mock_db.find_one = AsyncMock(return_value=None)

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Execute nonexistent report - should raise ValueError
        with pytest.raises(ValueError):
            await analytics_service.execute_report("nonexistent")

    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test database error handling"""
        from services.analytics_service import AnalyticsService
        from models.analytics_models import DashboardCreate

        # Create mock database that raises error
        mock_db = AsyncMock()
        mock_db.insert_one = AsyncMock(side_effect=Exception("Database error"))

        # Create analytics service
        analytics_service = AnalyticsService(mock_db)

        # Try to create dashboard - should raise exception
        dashboard_create = DashboardCreate(
            name="Test Dashboard",
            description="Test",
            owner_id="user_123",
            is_public=False
        )

        with pytest.raises(Exception):
            await analytics_service.create_dashboard(dashboard_create)
