"""
Advanced Analytics & Reporting Module Data Models for CPAS4
Cross-module analytics, dashboards, and reporting capabilities
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class ReportType(str, Enum):
    SALES = "sales"
    FINANCIAL = "financial"
    INVENTORY = "inventory"
    PROJECT = "project"
    CRM = "crm"
    ATS = "ats"
    AGENT_PERFORMANCE = "agent_performance"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "excel"


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    TABLE = "table"


class AggregationType(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class TimePeriod(str, Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_QUARTER = "this_quarter"
    LAST_QUARTER = "last_quarter"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


# Dashboard Models
class DashboardWidgetBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    widget_type: ChartType
    data_source: str  # Module name (crm, ats, finance, etc.)
    metric: str  # Metric to display
    aggregation: AggregationType = AggregationType.COUNT
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_period: TimePeriod = TimePeriod.THIS_MONTH
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    position: Dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 3})
    refresh_interval: int = Field(default=300, ge=0)  # seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardWidgetCreate(DashboardWidgetBase):
    pass


class DashboardWidgetUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    filters: Optional[Dict[str, Any]] = None
    time_period: Optional[TimePeriod] = None
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    position: Optional[Dict[str, int]] = None
    refresh_interval: Optional[int] = Field(None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class DashboardWidget(DashboardWidgetBase):
    widget_id: str
    dashboard_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime
    created_at: datetime
    updated_at: datetime


class DashboardBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    is_default: bool = False
    is_public: bool = False
    owner_id: str = "system"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DashboardCreate(DashboardBase):
    pass


class DashboardUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    is_default: Optional[bool] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Dashboard(DashboardBase):
    dashboard_id: str
    widget_count: int = 0
    created_at: datetime
    updated_at: datetime


# Report Models
class ReportFilterBase(BaseModel):
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, contains
    value: Any


class ReportBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    report_type: ReportType
    description: Optional[str] = None
    data_sources: List[str] = Field(..., min_items=1)  # Module names
    metrics: List[str] = Field(..., min_items=1)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[ReportFilterBase] = Field(default_factory=list)
    time_period: TimePeriod = TimePeriod.THIS_MONTH
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    format: ReportFormat = ReportFormat.JSON
    schedule: Optional[str] = None  # Cron expression
    recipients: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportCreate(ReportBase):
    pass


class ReportUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    filters: Optional[List[ReportFilterBase]] = None
    time_period: Optional[TimePeriod] = None
    custom_start_date: Optional[datetime] = None
    custom_end_date: Optional[datetime] = None
    format: Optional[ReportFormat] = None
    schedule: Optional[str] = None
    recipients: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Report(ReportBase):
    report_id: str
    last_run: Optional[datetime] = None
    run_count: int = 0
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"


# Report Execution Models
class ReportExecutionBase(BaseModel):
    report_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ReportExecutionCreate(ReportExecutionBase):
    pass


class ReportExecution(ReportExecutionBase):
    execution_id: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    row_count: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    file_path: Optional[str] = None


# Analytics Models
class MetricData(BaseModel):
    metric_name: str
    value: Union[int, float, str]
    previous_value: Optional[Union[int, float, str]] = None
    change_percent: Optional[float] = None
    trend: Optional[str] = None  # up, down, stable
    timestamp: datetime


class TimeSeriesData(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossModuleAnalytics(BaseModel):
    period_start: datetime
    period_end: datetime
    
    # CRM Metrics
    total_contacts: int = 0
    total_companies: int = 0
    total_deals: int = 0
    deals_value: float = 0.0
    
    # ATS Metrics
    total_candidates: int = 0
    total_jobs: int = 0
    total_applications: int = 0
    
    # Finance Metrics
    total_revenue: float = 0.0
    total_expenses: float = 0.0
    net_profit: float = 0.0
    
    # Project Metrics
    total_projects: int = 0
    active_projects: int = 0
    completed_projects: int = 0
    total_project_hours: float = 0.0
    
    # Inventory Metrics
    total_products: int = 0
    total_stock_value: float = 0.0
    low_stock_alerts: int = 0
    
    # Agent Metrics
    total_agents: int = 0
    total_conversations: int = 0
    avg_response_time: float = 0.0
    
    # Trends
    revenue_trend: List[Dict[str, Any]] = Field(default_factory=list)
    project_completion_trend: List[Dict[str, Any]] = Field(default_factory=list)
    sales_pipeline_trend: List[Dict[str, Any]] = Field(default_factory=list)


# KPI Models
class KPIBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    data_source: str
    metric: str
    target_value: float
    current_value: float = 0.0
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    is_higher_better: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KPICreate(KPIBase):
    pass


class KPIUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class KPI(KPIBase):
    kpi_id: str
    achievement_percent: float = 0.0
    status: str = "on_track"  # on_track, warning, critical
    last_updated: datetime
    created_at: datetime
    updated_at: datetime


# Statistics Models
class AnalyticsStats(BaseModel):
    total_dashboards: int
    total_reports: int
    total_widgets: int
    total_kpis: int
    total_report_executions: int
    recent_executions: int
    dashboards_by_owner: Dict[str, int]
    reports_by_type: Dict[str, int]
    kpis_by_status: Dict[str, int]
    most_used_reports: List[Dict[str, Any]]
    recent_reports: List[Dict[str, Any]]

