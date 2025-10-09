"""
Inventory Management Service for CPAS4 - Real Database Integration
Complete inventory management with products, stock, warehouses, and purchase orders
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.inventory_models import *

logger = logging.getLogger(__name__)

class InventoryService:
    def __init__(self, database: Database):
        self.db = database
        self.products_collection = "inventory_products"
        self.stock_collection = "inventory_stock"
        self.warehouses_collection = "inventory_warehouses"
        self.purchase_orders_collection = "inventory_purchase_orders"
        self.stock_movements_collection = "inventory_stock_movements"
        self.stock_alerts_collection = "inventory_stock_alerts"
        logger.info("✅ Inventory Service initialized with real database")
    
    # Product CRUD
    async def create_product(self, product_create: ProductCreate) -> Product:
        product_id = f"prd_{uuid.uuid4().hex[:12]}"
        product = Product(product_id=product_id, **product_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        pd = product.model_dump()
        for k in ["created_at", "updated_at"]:
            if pd.get(k): pd[k] = pd[k].isoformat()
        await self.db.insert_one(self.products_collection, pd)
        logger.info(f"✅ Product created: {product_id}")
        return product
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        pd = await self.db.find_one(self.products_collection, {"product_id": product_id})
        if not pd: return None
        for k in ["created_at", "updated_at"]:
            if k in pd and pd[k] and isinstance(pd[k], str): pd[k] = datetime.fromisoformat(pd[k])
        return Product(**pd)
    
    async def update_product(self, product_id: str, product_update: ProductUpdate) -> Optional[Product]:
        ud = product_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_product(product_id) if await self.db.update_one(self.products_collection, {"product_id": product_id}, {"$set": ud}) else None
    
    async def delete_product(self, product_id: str) -> bool:
        return await self.db.delete_one(self.products_collection, {"product_id": product_id})
    
    async def list_products(self, status: Optional[ProductStatus] = None, category: Optional[str] = None, limit: int = 100) -> List[Product]:
        f = {}
        if status: f["status"] = status
        if category: f["category"] = category
        pds = await self.db.find_many(self.products_collection, f, limit=limit)
        products = []
        for pd in pds:
            for k in ["created_at", "updated_at"]:
                if k in pd and pd[k] and isinstance(pd[k], str): pd[k] = datetime.fromisoformat(pd[k])
            products.append(Product(**pd))
        return products
    
    async def update_product_stock(self, product_id: str):
        stocks = await self.list_stock(product_id=product_id, limit=10000)
        total = sum(s.quantity for s in stocks)
        reserved = sum(s.reserved_quantity for s in stocks)
        available = total - reserved
        await self.db.update_one(self.products_collection, {"product_id": product_id}, {"$set": {"total_stock": total, "reserved_stock": reserved, "available_stock": available}})
        
        # Check for stock alerts
        product = await self.get_product(product_id)
        if product:
            if available == 0:
                await self.create_stock_alert(product_id, StockAlertType.OUT_OF_STOCK, available, 0)
            elif available <= product.reorder_point:
                await self.create_stock_alert(product_id, StockAlertType.LOW_STOCK, available, product.reorder_point)
    
    # Stock CRUD
    async def create_stock(self, stock_create: StockCreate) -> Stock:
        stock_id = f"stk_{uuid.uuid4().hex[:12]}"
        available = stock_create.quantity - stock_create.reserved_quantity
        stock = Stock(stock_id=stock_id, **stock_create.model_dump(), available_quantity=available, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        sd = stock.model_dump()
        for k in ["created_at", "updated_at", "expiry_date"]:
            if sd.get(k): sd[k] = sd[k].isoformat()
        await self.db.insert_one(self.stock_collection, sd)
        await self.update_product_stock(stock.product_id)
        logger.info(f"✅ Stock created: {stock_id}")
        return stock
    
    async def get_stock(self, stock_id: str) -> Optional[Stock]:
        sd = await self.db.find_one(self.stock_collection, {"stock_id": stock_id})
        if not sd: return None
        for k in ["created_at", "updated_at", "expiry_date"]:
            if k in sd and sd[k] and isinstance(sd[k], str): sd[k] = datetime.fromisoformat(sd[k])
        return Stock(**sd)
    
    async def update_stock(self, stock_id: str, stock_update: StockUpdate) -> Optional[Stock]:
        ud = stock_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "quantity" in ud or "reserved_quantity" in ud:
            stock = await self.get_stock(stock_id)
            if stock:
                qty = ud.get("quantity", stock.quantity)
                reserved = ud.get("reserved_quantity", stock.reserved_quantity)
                ud["available_quantity"] = qty - reserved
        stock = await self.get_stock(stock_id)
        if stock and await self.db.update_one(self.stock_collection, {"stock_id": stock_id}, {"$set": ud}):
            await self.update_product_stock(stock.product_id)
            return await self.get_stock(stock_id)
        return None
    
    async def delete_stock(self, stock_id: str) -> bool:
        stock = await self.get_stock(stock_id)
        if stock and await self.db.delete_one(self.stock_collection, {"stock_id": stock_id}):
            await self.update_product_stock(stock.product_id)
            return True
        return False
    
    async def list_stock(self, product_id: Optional[str] = None, warehouse_id: Optional[str] = None, limit: int = 100) -> List[Stock]:
        f = {}
        if product_id: f["product_id"] = product_id
        if warehouse_id: f["warehouse_id"] = warehouse_id
        sds = await self.db.find_many(self.stock_collection, f, limit=limit)
        stocks = []
        for sd in sds:
            for k in ["created_at", "updated_at", "expiry_date"]:
                if k in sd and sd[k] and isinstance(sd[k], str): sd[k] = datetime.fromisoformat(sd[k])
            stocks.append(Stock(**sd))
        return stocks
    
    # Warehouse CRUD
    async def create_warehouse(self, warehouse_create: WarehouseCreate) -> Warehouse:
        warehouse_id = f"wh_{uuid.uuid4().hex[:12]}"
        warehouse = Warehouse(warehouse_id=warehouse_id, **warehouse_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        wd = warehouse.model_dump()
        for k in ["created_at", "updated_at"]:
            if wd.get(k): wd[k] = wd[k].isoformat()
        await self.db.insert_one(self.warehouses_collection, wd)
        logger.info(f"✅ Warehouse created: {warehouse_id}")
        return warehouse
    
    async def get_warehouse(self, warehouse_id: str) -> Optional[Warehouse]:
        wd = await self.db.find_one(self.warehouses_collection, {"warehouse_id": warehouse_id})
        if not wd: return None
        for k in ["created_at", "updated_at"]:
            if k in wd and wd[k] and isinstance(wd[k], str): wd[k] = datetime.fromisoformat(wd[k])
        return Warehouse(**wd)
    
    async def update_warehouse(self, warehouse_id: str, warehouse_update: WarehouseUpdate) -> Optional[Warehouse]:
        ud = warehouse_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_warehouse(warehouse_id) if await self.db.update_one(self.warehouses_collection, {"warehouse_id": warehouse_id}, {"$set": ud}) else None
    
    async def delete_warehouse(self, warehouse_id: str) -> bool:
        return await self.db.delete_one(self.warehouses_collection, {"warehouse_id": warehouse_id})
    
    async def list_warehouses(self, is_active: Optional[bool] = None, limit: int = 100) -> List[Warehouse]:
        f = {}
        if is_active is not None: f["is_active"] = is_active
        wds = await self.db.find_many(self.warehouses_collection, f, limit=limit)
        warehouses = []
        for wd in wds:
            for k in ["created_at", "updated_at"]:
                if k in wd and wd[k] and isinstance(wd[k], str): wd[k] = datetime.fromisoformat(wd[k])
            warehouses.append(Warehouse(**wd))
        return warehouses
    
    # Purchase Order CRUD
    async def create_purchase_order(self, po_create: PurchaseOrderCreate) -> PurchaseOrder:
        order_id = f"po_{uuid.uuid4().hex[:12]}"
        total = sum(item.total_price for item in po_create.line_items)
        po = PurchaseOrder(order_id=order_id, **po_create.model_dump(), total_amount=total, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        pod = po.model_dump()
        for k in ["created_at", "updated_at", "order_date", "expected_delivery_date", "approved_at", "received_at"]:
            if pod.get(k): pod[k] = pod[k].isoformat()
        await self.db.insert_one(self.purchase_orders_collection, pod)
        logger.info(f"✅ Purchase order created: {order_id}")
        return po
    
    async def get_purchase_order(self, order_id: str) -> Optional[PurchaseOrder]:
        pod = await self.db.find_one(self.purchase_orders_collection, {"order_id": order_id})
        if not pod: return None
        for k in ["created_at", "updated_at", "order_date", "expected_delivery_date", "approved_at", "received_at"]:
            if k in pod and pod[k] and isinstance(pod[k], str): pod[k] = datetime.fromisoformat(pod[k])
        return PurchaseOrder(**pod)
    
    async def update_purchase_order(self, order_id: str, po_update: PurchaseOrderUpdate) -> Optional[PurchaseOrder]:
        ud = po_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "status" in ud:
            if ud["status"] == PurchaseOrderStatus.APPROVED:
                po = await self.get_purchase_order(order_id)
                if po and not po.approved_at: ud["approved_at"] = datetime.utcnow().isoformat()
            elif ud["status"] == PurchaseOrderStatus.RECEIVED:
                po = await self.get_purchase_order(order_id)
                if po and not po.received_at:
                    ud["received_at"] = datetime.utcnow().isoformat()
                    ud["received_amount"] = po.total_amount
        if "expected_delivery_date" in ud and ud["expected_delivery_date"]: ud["expected_delivery_date"] = ud["expected_delivery_date"].isoformat()
        return await self.get_purchase_order(order_id) if await self.db.update_one(self.purchase_orders_collection, {"order_id": order_id}, {"$set": ud}) else None
    
    async def delete_purchase_order(self, order_id: str) -> bool:
        return await self.db.delete_one(self.purchase_orders_collection, {"order_id": order_id})
    
    async def list_purchase_orders(self, status: Optional[PurchaseOrderStatus] = None, warehouse_id: Optional[str] = None, limit: int = 100) -> List[PurchaseOrder]:
        f = {}
        if status: f["status"] = status
        if warehouse_id: f["warehouse_id"] = warehouse_id
        pods = await self.db.find_many(self.purchase_orders_collection, f, limit=limit)
        orders = []
        for pod in pods:
            for k in ["created_at", "updated_at", "order_date", "expected_delivery_date", "approved_at", "received_at"]:
                if k in pod and pod[k] and isinstance(pod[k], str): pod[k] = datetime.fromisoformat(pod[k])
            orders.append(PurchaseOrder(**pod))
        return orders
    
    # Stock Movement
    async def create_stock_movement(self, movement_create: StockMovementCreate) -> StockMovement:
        movement_id = f"mv_{uuid.uuid4().hex[:12]}"
        movement = StockMovement(movement_id=movement_id, **movement_create.model_dump(), created_at=datetime.utcnow())
        md = movement.model_dump()
        md["created_at"] = md["created_at"].isoformat()
        await self.db.insert_one(self.stock_movements_collection, md)
        logger.info(f"✅ Stock movement created: {movement_id}")
        return movement
    
    async def list_stock_movements(self, product_id: Optional[str] = None, warehouse_id: Optional[str] = None, limit: int = 100) -> List[StockMovement]:
        f = {}
        if product_id: f["product_id"] = product_id
        if warehouse_id: f["$or"] = [{"from_warehouse_id": warehouse_id}, {"to_warehouse_id": warehouse_id}]
        mds = await self.db.find_many(self.stock_movements_collection, f, limit=limit)
        movements = []
        for md in mds:
            if "created_at" in md and isinstance(md["created_at"], str): md["created_at"] = datetime.fromisoformat(md["created_at"])
            movements.append(StockMovement(**md))
        return movements
    
    # Stock Alerts
    async def create_stock_alert(self, product_id: str, alert_type: StockAlertType, current_qty: int, threshold_qty: int):
        # Check if alert already exists
        existing = await self.db.find_one(self.stock_alerts_collection, {"product_id": product_id, "alert_type": alert_type, "is_resolved": False})
        if existing: return
        
        alert_id = f"alt_{uuid.uuid4().hex[:12]}"
        message = f"{alert_type.value.replace('_', ' ').title()}: {current_qty} units (threshold: {threshold_qty})"
        alert = StockAlert(alert_id=alert_id, product_id=product_id, warehouse_id="default", alert_type=alert_type, current_quantity=current_qty, threshold_quantity=threshold_qty, message=message, created_at=datetime.utcnow())
        ad = alert.model_dump()
        for k in ["created_at", "resolved_at"]:
            if ad.get(k): ad[k] = ad[k].isoformat()
        await self.db.insert_one(self.stock_alerts_collection, ad)
    
    async def list_stock_alerts(self, is_resolved: Optional[bool] = None, limit: int = 100) -> List[StockAlert]:
        f = {}
        if is_resolved is not None: f["is_resolved"] = is_resolved
        ads = await self.db.find_many(self.stock_alerts_collection, f, limit=limit)
        alerts = []
        for ad in ads:
            for k in ["created_at", "resolved_at"]:
                if k in ad and ad[k] and isinstance(ad[k], str): ad[k] = datetime.fromisoformat(ad[k])
            alerts.append(StockAlert(**ad))
        return alerts
    
    # Statistics
    async def get_stats(self) -> InventoryStats:
        all_products = await self.list_products(limit=10000)
        all_stock = await self.list_stock(limit=10000)
        all_warehouses = await self.list_warehouses(limit=10000)
        all_pos = await self.list_purchase_orders(limit=10000)
        all_alerts = await self.list_stock_alerts(limit=10000)
        all_movements = await self.list_stock_movements(limit=10000)
        
        products_by_status = {s.value: sum(1 for p in all_products if p.status == s) for s in ProductStatus}
        po_by_status = {s.value: sum(1 for po in all_pos if po.status == s) for s in PurchaseOrderStatus}
        alerts_by_type = {t.value: sum(1 for a in all_alerts if a.alert_type == t) for t in StockAlertType}
        movements_by_type = {t.value: sum(1 for m in all_movements if m.movement_type == t) for t in StockMovementType}
        
        total_value = sum(p.total_stock * p.cost_price for p in all_products)
        low_stock = sum(1 for p in all_products if p.available_stock <= p.reorder_point and p.available_stock > 0)
        out_of_stock = sum(1 for p in all_products if p.available_stock == 0)
        
        top_products = sorted(all_products, key=lambda p: p.total_stock * p.cost_price, reverse=True)[:5]
        top_products_data = [{"product_id": p.product_id, "name": p.name, "value": round(p.total_stock * p.cost_price, 2)} for p in top_products]
        
        return InventoryStats(
            total_products=len(all_products), active_products=sum(1 for p in all_products if p.status == ProductStatus.ACTIVE),
            total_warehouses=len(all_warehouses), total_stock_items=len(all_stock),
            total_stock_value=round(total_value, 2), total_purchase_orders=len(all_pos),
            pending_purchase_orders=sum(1 for po in all_pos if po.status in [PurchaseOrderStatus.PENDING, PurchaseOrderStatus.APPROVED]),
            total_stock_alerts=len(all_alerts), unresolved_alerts=sum(1 for a in all_alerts if not a.is_resolved),
            products_by_status=products_by_status, stock_movements_by_type=movements_by_type,
            purchase_orders_by_status=po_by_status, alerts_by_type=alerts_by_type,
            low_stock_products=low_stock, out_of_stock_products=out_of_stock,
            top_products_by_value=top_products_data, warehouse_utilization={}
        )

