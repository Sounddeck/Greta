"""
Inventory Management Module Data Models for CPAS4
Complete inventory management with products, stock, warehouses, and purchase orders
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class ProductStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class StockMovementType(str, Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"
    RETURN = "return"
    DAMAGE = "damage"


class PurchaseOrderStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    ORDERED = "ordered"
    RECEIVED = "received"
    CANCELLED = "cancelled"


class StockAlertType(str, Enum):
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    OVERSTOCK = "overstock"
    EXPIRING_SOON = "expiring_soon"


# Product Models
class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    sku: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    unit_price: float = Field(..., ge=0)
    cost_price: float = Field(..., ge=0)
    currency: str = Field(default="USD", max_length=3)
    unit_of_measure: str = Field(default="unit")
    barcode: Optional[str] = None
    weight: Optional[float] = Field(None, ge=0)
    dimensions: Optional[Dict[str, float]] = None
    reorder_point: int = Field(default=10, ge=0)
    reorder_quantity: int = Field(default=50, ge=0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[ProductStatus] = None
    unit_price: Optional[float] = Field(None, ge=0)
    cost_price: Optional[float] = Field(None, ge=0)
    reorder_point: Optional[int] = Field(None, ge=0)
    reorder_quantity: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Product(ProductBase):
    product_id: str
    status: ProductStatus = ProductStatus.ACTIVE
    total_stock: int = 0
    available_stock: int = 0
    reserved_stock: int = 0
    created_at: datetime
    updated_at: datetime


# Stock Models
class StockBase(BaseModel):
    product_id: str
    warehouse_id: str
    quantity: int = Field(..., ge=0)
    reserved_quantity: int = Field(default=0, ge=0)
    location: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StockCreate(StockBase):
    pass


class StockUpdate(BaseModel):
    quantity: Optional[int] = Field(None, ge=0)
    reserved_quantity: Optional[int] = Field(None, ge=0)
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Stock(StockBase):
    stock_id: str
    available_quantity: int = 0
    created_at: datetime
    updated_at: datetime


# Stock Movement Models
class StockMovementBase(BaseModel):
    product_id: str
    movement_type: StockMovementType
    quantity: int = Field(..., gt=0)
    from_warehouse_id: Optional[str] = None
    to_warehouse_id: Optional[str] = None
    reference_id: Optional[str] = None
    reference_type: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StockMovementCreate(StockMovementBase):
    pass


class StockMovement(StockMovementBase):
    movement_id: str
    created_at: datetime
    created_by: str = "system"


# Warehouse Models
class WarehouseBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=50)
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    manager_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WarehouseCreate(WarehouseBase):
    pass


class WarehouseUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    address: Optional[str] = None
    is_active: Optional[bool] = None
    manager_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Warehouse(WarehouseBase):
    warehouse_id: str
    total_products: int = 0
    total_stock_value: float = 0.0
    created_at: datetime
    updated_at: datetime


# Purchase Order Models
class PurchaseOrderLineItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    
    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price


class PurchaseOrderBase(BaseModel):
    order_number: str = Field(..., min_length=1, max_length=50)
    supplier_id: Optional[str] = None
    supplier_name: str = Field(..., min_length=1, max_length=200)
    warehouse_id: str
    line_items: List[PurchaseOrderLineItem] = Field(..., min_items=1)
    order_date: datetime = Field(default_factory=datetime.utcnow)
    expected_delivery_date: Optional[datetime] = None
    notes: Optional[str] = None
    currency: str = Field(default="USD", max_length=3)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PurchaseOrderCreate(PurchaseOrderBase):
    pass


class PurchaseOrderUpdate(BaseModel):
    supplier_name: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[PurchaseOrderStatus] = None
    expected_delivery_date: Optional[datetime] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PurchaseOrder(PurchaseOrderBase):
    order_id: str
    status: PurchaseOrderStatus = PurchaseOrderStatus.DRAFT
    total_amount: float = 0.0
    received_amount: float = 0.0
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    created_by: str = "system"


# Stock Alert Models
class StockAlertBase(BaseModel):
    product_id: str
    warehouse_id: str
    alert_type: StockAlertType
    current_quantity: int
    threshold_quantity: int
    message: str
    is_resolved: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StockAlertCreate(StockAlertBase):
    pass


class StockAlert(StockAlertBase):
    alert_id: str
    created_at: datetime
    resolved_at: Optional[datetime] = None


# Statistics Models
class InventoryStats(BaseModel):
    total_products: int
    active_products: int
    total_warehouses: int
    total_stock_items: int
    total_stock_value: float
    total_purchase_orders: int
    pending_purchase_orders: int
    total_stock_alerts: int
    unresolved_alerts: int
    products_by_status: Dict[str, int]
    stock_movements_by_type: Dict[str, int]
    purchase_orders_by_status: Dict[str, int]
    alerts_by_type: Dict[str, int]
    low_stock_products: int
    out_of_stock_products: int
    top_products_by_value: List[Dict[str, Any]]
    warehouse_utilization: Dict[str, float]


# Report Models
class InventoryReport(BaseModel):
    report_type: str
    warehouse_id: Optional[str] = None
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any]

