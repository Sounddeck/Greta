"""
Inventory Management Router for CPAS4
REST API endpoints for Inventory Management
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.inventory_service import InventoryService
from models.inventory_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/inventory", tags=["Inventory"])

async def get_service() -> InventoryService:
    db = Database()
    await db.connect()
    return InventoryService(db)

# Product endpoints
@router.post("/products", response_model=Product, status_code=201)
async def create_product(product: ProductCreate, svc: InventoryService = Depends(get_service)):
    return await svc.create_product(product)

@router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str, svc: InventoryService = Depends(get_service)):
    prd = await svc.get_product(product_id)
    if not prd: raise HTTPException(404, "Product not found")
    return prd

@router.patch("/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product_update: ProductUpdate, svc: InventoryService = Depends(get_service)):
    prd = await svc.update_product(product_id, product_update)
    if not prd: raise HTTPException(404, "Product not found")
    return prd

@router.delete("/products/{product_id}", status_code=204)
async def delete_product(product_id: str, svc: InventoryService = Depends(get_service)):
    if not await svc.delete_product(product_id): raise HTTPException(404, "Product not found")

@router.get("/products", response_model=List[Product])
async def list_products(status: Optional[ProductStatus] = None, category: Optional[str] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_products(status, category, limit)

# Stock endpoints
@router.post("/stock", response_model=Stock, status_code=201)
async def create_stock(stock: StockCreate, svc: InventoryService = Depends(get_service)):
    return await svc.create_stock(stock)

@router.get("/stock/{stock_id}", response_model=Stock)
async def get_stock(stock_id: str, svc: InventoryService = Depends(get_service)):
    stk = await svc.get_stock(stock_id)
    if not stk: raise HTTPException(404, "Stock not found")
    return stk

@router.patch("/stock/{stock_id}", response_model=Stock)
async def update_stock(stock_id: str, stock_update: StockUpdate, svc: InventoryService = Depends(get_service)):
    stk = await svc.update_stock(stock_id, stock_update)
    if not stk: raise HTTPException(404, "Stock not found")
    return stk

@router.delete("/stock/{stock_id}", status_code=204)
async def delete_stock(stock_id: str, svc: InventoryService = Depends(get_service)):
    if not await svc.delete_stock(stock_id): raise HTTPException(404, "Stock not found")

@router.get("/stock", response_model=List[Stock])
async def list_stock(product_id: Optional[str] = None, warehouse_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_stock(product_id, warehouse_id, limit)

# Warehouse endpoints
@router.post("/warehouses", response_model=Warehouse, status_code=201)
async def create_warehouse(warehouse: WarehouseCreate, svc: InventoryService = Depends(get_service)):
    return await svc.create_warehouse(warehouse)

@router.get("/warehouses/{warehouse_id}", response_model=Warehouse)
async def get_warehouse(warehouse_id: str, svc: InventoryService = Depends(get_service)):
    wh = await svc.get_warehouse(warehouse_id)
    if not wh: raise HTTPException(404, "Warehouse not found")
    return wh

@router.patch("/warehouses/{warehouse_id}", response_model=Warehouse)
async def update_warehouse(warehouse_id: str, warehouse_update: WarehouseUpdate, svc: InventoryService = Depends(get_service)):
    wh = await svc.update_warehouse(warehouse_id, warehouse_update)
    if not wh: raise HTTPException(404, "Warehouse not found")
    return wh

@router.delete("/warehouses/{warehouse_id}", status_code=204)
async def delete_warehouse(warehouse_id: str, svc: InventoryService = Depends(get_service)):
    if not await svc.delete_warehouse(warehouse_id): raise HTTPException(404, "Warehouse not found")

@router.get("/warehouses", response_model=List[Warehouse])
async def list_warehouses(is_active: Optional[bool] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_warehouses(is_active, limit)

# Purchase Order endpoints
@router.post("/purchase-orders", response_model=PurchaseOrder, status_code=201)
async def create_purchase_order(po: PurchaseOrderCreate, svc: InventoryService = Depends(get_service)):
    return await svc.create_purchase_order(po)

@router.get("/purchase-orders/{order_id}", response_model=PurchaseOrder)
async def get_purchase_order(order_id: str, svc: InventoryService = Depends(get_service)):
    po = await svc.get_purchase_order(order_id)
    if not po: raise HTTPException(404, "Purchase order not found")
    return po

@router.patch("/purchase-orders/{order_id}", response_model=PurchaseOrder)
async def update_purchase_order(order_id: str, po_update: PurchaseOrderUpdate, svc: InventoryService = Depends(get_service)):
    po = await svc.update_purchase_order(order_id, po_update)
    if not po: raise HTTPException(404, "Purchase order not found")
    return po

@router.delete("/purchase-orders/{order_id}", status_code=204)
async def delete_purchase_order(order_id: str, svc: InventoryService = Depends(get_service)):
    if not await svc.delete_purchase_order(order_id): raise HTTPException(404, "Purchase order not found")

@router.get("/purchase-orders", response_model=List[PurchaseOrder])
async def list_purchase_orders(status: Optional[PurchaseOrderStatus] = None, warehouse_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_purchase_orders(status, warehouse_id, limit)

# Stock Movement endpoints
@router.post("/stock-movements", response_model=StockMovement, status_code=201)
async def create_stock_movement(movement: StockMovementCreate, svc: InventoryService = Depends(get_service)):
    return await svc.create_stock_movement(movement)

@router.get("/stock-movements", response_model=List[StockMovement])
async def list_stock_movements(product_id: Optional[str] = None, warehouse_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_stock_movements(product_id, warehouse_id, limit)

# Stock Alerts endpoints
@router.get("/stock-alerts", response_model=List[StockAlert])
async def list_stock_alerts(is_resolved: Optional[bool] = None, limit: int = Query(100, le=1000), svc: InventoryService = Depends(get_service)):
    return await svc.list_stock_alerts(is_resolved, limit)

# Statistics
@router.get("/stats", response_model=InventoryStats)
async def get_stats(svc: InventoryService = Depends(get_service)):
    return await svc.get_stats()

# Health check
@router.get("/health")
async def health_check():
    return {"status": "healthy", "module": "Inventory", "features": ["Product Management", "Stock Tracking", "Warehouse Management", "Purchase Orders", "Stock Movements", "Stock Alerts", "Statistics"], "endpoints": 27, "database": "MongoDB/File Storage"}

