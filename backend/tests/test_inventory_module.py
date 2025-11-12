"""
Inventory Management Module Test Suite
Comprehensive tests for Inventory Management functionality
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from database import Database
from services.inventory_service import InventoryService
from models.inventory_models import *

class TestInventoryModule:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
    async def setup(self):
        """Initialize database and service"""
        self.db = Database()
        await self.db.connect()
        self.service = InventoryService(self.db)
        print("✅ Inventory module test setup complete")
    
    async def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            for collection in ["inventory_products", "inventory_stock", "inventory_warehouses", "inventory_purchase_orders", "inventory_stock_movements", "inventory_stock_alerts"]:
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
    
    async def test_product_crud(self):
        """Test product CRUD operations"""
        try:
            # Create product
            product_create = ProductCreate(
                name="Test Product",
                sku="TST-001",
                description="A test product",
                category="Electronics",
                brand="TestBrand",
                unit_price=99.99,
                cost_price=50.00,
                reorder_point=10,
                reorder_quantity=50
            )
            product = await self.service.create_product(product_create)
            assert product.product_id.startswith("prd_"), "Product ID format incorrect"
            assert product.name == "Test Product", "Product name mismatch"
            
            # Get product
            retrieved = await self.service.get_product(product.product_id)
            assert retrieved is not None, "Product not found"
            assert retrieved.product_id == product.product_id, "Product ID mismatch"
            
            # Update product
            update = ProductUpdate(name="Updated Product", status=ProductStatus.ACTIVE)
            updated = await self.service.update_product(product.product_id, update)
            assert updated is not None, "Product update failed"
            assert updated.name == "Updated Product", "Product name not updated"
            
            # List products
            products = await self.service.list_products()
            assert len(products) >= 1, "No products found"
            
            # Delete product
            deleted = await self.service.delete_product(product.product_id)
            assert deleted, "Product deletion failed"
            
            self.record_result("Product CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Product CRUD", False, str(e))
    
    async def test_stock_crud(self):
        """Test stock CRUD operations"""
        try:
            # Create product and warehouse first
            product_create = ProductCreate(name="Stock Test Product", sku="STK-001", unit_price=10.00, cost_price=5.00)
            product = await self.service.create_product(product_create)
            
            warehouse_create = WarehouseCreate(name="Test Warehouse", code="WH-001")
            warehouse = await self.service.create_warehouse(warehouse_create)
            
            # Create stock
            stock_create = StockCreate(
                product_id=product.product_id,
                warehouse_id=warehouse.warehouse_id,
                quantity=100,
                reserved_quantity=10,
                location="A1-B2"
            )
            stock = await self.service.create_stock(stock_create)
            assert stock.stock_id.startswith("stk_"), "Stock ID format incorrect"
            assert stock.available_quantity == 90, "Available quantity calculation incorrect"
            
            # Get stock
            retrieved = await self.service.get_stock(stock.stock_id)
            assert retrieved is not None, "Stock not found"
            
            # Update stock
            update = StockUpdate(quantity=150, reserved_quantity=20)
            updated = await self.service.update_stock(stock.stock_id, update)
            assert updated is not None, "Stock update failed"
            assert updated.available_quantity == 130, "Available quantity not updated correctly"
            
            # List stock
            stocks = await self.service.list_stock(product_id=product.product_id)
            assert len(stocks) >= 1, "No stock found"
            
            self.record_result("Stock CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Stock CRUD", False, str(e))
    
    async def test_warehouse_crud(self):
        """Test warehouse CRUD operations"""
        try:
            # Create warehouse
            warehouse_create = WarehouseCreate(
                name="Main Warehouse",
                code="WH-MAIN",
                address="123 Main St",
                city="New York",
                state="NY",
                country="USA",
                manager_name="John Doe"
            )
            warehouse = await self.service.create_warehouse(warehouse_create)
            assert warehouse.warehouse_id.startswith("wh_"), "Warehouse ID format incorrect"
            assert warehouse.name == "Main Warehouse", "Warehouse name mismatch"
            
            # Get warehouse
            retrieved = await self.service.get_warehouse(warehouse.warehouse_id)
            assert retrieved is not None, "Warehouse not found"
            
            # Update warehouse
            update = WarehouseUpdate(manager_name="Jane Smith", is_active=True)
            updated = await self.service.update_warehouse(warehouse.warehouse_id, update)
            assert updated is not None, "Warehouse update failed"
            assert updated.manager_name == "Jane Smith", "Manager name not updated"
            
            # List warehouses
            warehouses = await self.service.list_warehouses()
            assert len(warehouses) >= 1, "No warehouses found"
            
            self.record_result("Warehouse CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Warehouse CRUD", False, str(e))
    
    async def test_purchase_order_crud(self):
        """Test purchase order CRUD operations"""
        try:
            # Create product and warehouse first
            product_create = ProductCreate(name="PO Test Product", sku="PO-001", unit_price=20.00, cost_price=10.00)
            product = await self.service.create_product(product_create)
            
            warehouse_create = WarehouseCreate(name="PO Warehouse", code="WH-PO")
            warehouse = await self.service.create_warehouse(warehouse_create)
            
            # Create purchase order
            line_items = [
                PurchaseOrderLineItem(product_id=product.product_id, product_name=product.name, quantity=50, unit_price=10.00)
            ]
            po_create = PurchaseOrderCreate(
                order_number="PO-001",
                supplier_name="Test Supplier",
                warehouse_id=warehouse.warehouse_id,
                line_items=line_items,
                expected_delivery_date=datetime.utcnow() + timedelta(days=7)
            )
            po = await self.service.create_purchase_order(po_create)
            assert po.order_id.startswith("po_"), "Purchase order ID format incorrect"
            assert po.total_amount == 500.00, "Total amount calculation incorrect"
            
            # Get purchase order
            retrieved = await self.service.get_purchase_order(po.order_id)
            assert retrieved is not None, "Purchase order not found"
            
            # Update purchase order
            update = PurchaseOrderUpdate(status=PurchaseOrderStatus.APPROVED)
            updated = await self.service.update_purchase_order(po.order_id, update)
            assert updated is not None, "Purchase order update failed"
            assert updated.status == PurchaseOrderStatus.APPROVED, "Status not updated"
            
            # List purchase orders
            pos = await self.service.list_purchase_orders()
            assert len(pos) >= 1, "No purchase orders found"
            
            self.record_result("Purchase Order CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Purchase Order CRUD", False, str(e))
    
    async def test_stock_movement(self):
        """Test stock movement tracking"""
        try:
            # Create product
            product_create = ProductCreate(name="Movement Test Product", sku="MV-001", unit_price=15.00, cost_price=7.50)
            product = await self.service.create_product(product_create)
            
            # Create stock movement
            movement_create = StockMovementCreate(
                product_id=product.product_id,
                movement_type=StockMovementType.PURCHASE,
                quantity=100,
                notes="Initial stock purchase"
            )
            movement = await self.service.create_stock_movement(movement_create)
            assert movement.movement_id.startswith("mv_"), "Movement ID format incorrect"
            assert movement.quantity == 100, "Movement quantity mismatch"
            
            # List movements
            movements = await self.service.list_stock_movements(product_id=product.product_id)
            assert len(movements) >= 1, "No movements found"
            
            self.record_result("Stock Movement", True, "Movement tracking successful")
        except Exception as e:
            self.record_result("Stock Movement", False, str(e))
    
    async def test_stock_alerts(self):
        """Test stock alert generation"""
        try:
            # Create product with low reorder point
            product_create = ProductCreate(name="Alert Test Product", sku="ALT-001", unit_price=10.00, cost_price=5.00, reorder_point=50)
            product = await self.service.create_product(product_create)
            
            warehouse_create = WarehouseCreate(name="Alert Warehouse", code="WH-ALT")
            warehouse = await self.service.create_warehouse(warehouse_create)
            
            # Create stock below reorder point
            stock_create = StockCreate(
                product_id=product.product_id,
                warehouse_id=warehouse.warehouse_id,
                quantity=30,
                reserved_quantity=0
            )
            await self.service.create_stock(stock_create)
            
            # Check alerts were created
            alerts = await self.service.list_stock_alerts(is_resolved=False)
            assert len(alerts) >= 1, "No alerts generated"
            
            self.record_result("Stock Alerts", True, "Alert generation successful")
        except Exception as e:
            self.record_result("Stock Alerts", False, str(e))
    
    async def test_statistics(self):
        """Test inventory statistics"""
        try:
            stats = await self.service.get_stats()
            assert stats is not None, "Stats not returned"
            assert hasattr(stats, 'total_products'), "Missing total_products"
            assert hasattr(stats, 'total_warehouses'), "Missing total_warehouses"
            assert hasattr(stats, 'total_stock_value'), "Missing total_stock_value"
            
            self.record_result("Statistics", True, f"Stats: {stats.total_products} products, {stats.total_warehouses} warehouses, ${stats.total_stock_value} value")
        except Exception as e:
            self.record_result("Statistics", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 INVENTORY MANAGEMENT MODULE TEST SUITE")
        print("="*80 + "\n")
        
        await self.setup()
        
        # Run tests
        await self.test_product_crud()
        await self.test_stock_crud()
        await self.test_warehouse_crud()
        await self.test_purchase_order_crud()
        await self.test_stock_movement()
        await self.test_stock_alerts()
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
    tester = TestInventoryModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

