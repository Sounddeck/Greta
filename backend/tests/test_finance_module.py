"""
Finance Module Test Suite
Comprehensive tests for Finance Management functionality
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from database import Database
from services.finance_service import FinanceService
from models.finance_models import *

class TestFinanceModule:
    def __init__(self):
        self.db = None
        self.service = None
        self.test_results = []
        
    async def setup(self):
        """Initialize database and service"""
        self.db = Database()
        await self.db.connect()
        self.service = FinanceService(self.db)
        print("✅ Finance module test setup complete")
    
    async def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            # Delete all test data
            for collection in ["finance_accounts", "finance_transactions", "finance_invoices", "finance_budgets"]:
                try:
                    # Get all documents and delete them one by one
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
    
    async def test_account_crud(self):
        """Test account CRUD operations"""
        try:
            # Create account
            account_create = AccountCreate(
                name="Business Checking",
                account_number="CHK-001",
                account_type=AccountType.ASSET,
                account_subtype=AccountSubType.BANK,
                description="Main business checking account",
                currency="USD"
            )
            account = await self.service.create_account(account_create)
            assert account.account_id.startswith("acc_"), "Account ID format incorrect"
            assert account.name == "Business Checking", "Account name mismatch"
            
            # Get account
            retrieved = await self.service.get_account(account.account_id)
            assert retrieved is not None, "Account not found"
            assert retrieved.account_id == account.account_id, "Account ID mismatch"
            
            # Update account
            update = AccountUpdate(name="Updated Checking", description="Updated description")
            updated = await self.service.update_account(account.account_id, update)
            assert updated is not None, "Account update failed"
            assert updated.name == "Updated Checking", "Account name not updated"
            
            # List accounts
            accounts = await self.service.list_accounts()
            assert len(accounts) >= 1, "No accounts found"
            
            # Delete account
            deleted = await self.service.delete_account(account.account_id)
            assert deleted, "Account deletion failed"
            
            self.record_result("Account CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Account CRUD", False, str(e))
    
    async def test_transaction_crud(self):
        """Test transaction CRUD operations"""
        try:
            # Create account first
            account_create = AccountCreate(
                name="Revenue Account",
                account_number="REV-001",
                account_type=AccountType.REVENUE,
                account_subtype=AccountSubType.SALES,
                currency="USD"
            )
            account = await self.service.create_account(account_create)
            
            # Create transaction
            transaction_create = TransactionCreate(
                transaction_type=TransactionType.INCOME,
                amount=1000.00,
                from_account_id=account.account_id,
                category="Sales",
                description="Product sale",
                payment_method=PaymentMethod.CREDIT_CARD
            )
            transaction = await self.service.create_transaction(transaction_create)
            assert transaction.transaction_id.startswith("txn_"), "Transaction ID format incorrect"
            assert transaction.amount == 1000.00, "Transaction amount mismatch"
            
            # Get transaction
            retrieved = await self.service.get_transaction(transaction.transaction_id)
            assert retrieved is not None, "Transaction not found"
            
            # Update transaction
            update = TransactionUpdate(description="Updated sale", status=TransactionStatus.CLEARED)
            updated = await self.service.update_transaction(transaction.transaction_id, update)
            assert updated is not None, "Transaction update failed"
            assert updated.status == TransactionStatus.CLEARED, "Transaction status not updated"
            
            # List transactions
            transactions = await self.service.list_transactions()
            assert len(transactions) >= 1, "No transactions found"
            
            self.record_result("Transaction CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Transaction CRUD", False, str(e))
    
    async def test_invoice_crud(self):
        """Test invoice CRUD operations"""
        try:
            # Create invoice
            line_items = [
                InvoiceLineItem(description="Product A", quantity=2, unit_price=50.00, tax_rate=10.0),
                InvoiceLineItem(description="Product B", quantity=1, unit_price=100.00, tax_rate=10.0)
            ]
            invoice_create = InvoiceCreate(
                invoice_number="INV-001",
                customer_name="Acme Corp",
                customer_email="billing@acme.com",
                line_items=line_items,
                issue_date=datetime.utcnow(),
                due_date=datetime.utcnow() + timedelta(days=30),
                payment_terms="Net 30"
            )
            invoice = await self.service.create_invoice(invoice_create)
            assert invoice.invoice_id.startswith("inv_"), "Invoice ID format incorrect"
            assert invoice.total_amount > 0, "Invoice total not calculated"
            assert len(invoice.line_items) == 2, "Line items count mismatch"
            
            # Get invoice
            retrieved = await self.service.get_invoice(invoice.invoice_id)
            assert retrieved is not None, "Invoice not found"
            
            # Update invoice
            update = InvoiceUpdate(status=InvoiceStatus.SENT, notes="Sent to customer")
            updated = await self.service.update_invoice(invoice.invoice_id, update)
            assert updated is not None, "Invoice update failed"
            assert updated.status == InvoiceStatus.SENT, "Invoice status not updated"
            
            # List invoices
            invoices = await self.service.list_invoices()
            assert len(invoices) >= 1, "No invoices found"
            
            self.record_result("Invoice CRUD", True, "All operations successful")
        except Exception as e:
            self.record_result("Invoice CRUD", False, str(e))
    
    async def test_account_balance_updates(self):
        """Test account balance updates from transactions"""
        try:
            # Create account
            account_create = AccountCreate(
                name="Test Account",
                account_number="TST-001",
                account_type=AccountType.ASSET,
                account_subtype=AccountSubType.CASH,
                currency="USD"
            )
            account = await self.service.create_account(account_create)
            initial_balance = account.balance
            
            # Create income transaction
            transaction_create = TransactionCreate(
                transaction_type=TransactionType.INCOME,
                amount=500.00,
                from_account_id=account.account_id,
                description="Test income"
            )
            await self.service.create_transaction(transaction_create)
            
            # Check balance updated
            updated_account = await self.service.get_account(account.account_id)
            assert updated_account.balance == initial_balance + 500.00, "Balance not updated correctly"
            
            self.record_result("Account Balance Updates", True, "Balance updated correctly")
        except Exception as e:
            self.record_result("Account Balance Updates", False, str(e))
    
    async def test_invoice_calculations(self):
        """Test invoice total calculations"""
        try:
            line_items = [
                InvoiceLineItem(description="Item 1", quantity=2, unit_price=100.00, tax_rate=10.0, discount=10.00),
                InvoiceLineItem(description="Item 2", quantity=1, unit_price=50.00, tax_rate=10.0)
            ]
            invoice_create = InvoiceCreate(
                invoice_number="INV-CALC-001",
                customer_name="Test Customer",
                line_items=line_items,
                issue_date=datetime.utcnow(),
                due_date=datetime.utcnow() + timedelta(days=30),
                discount=20.00
            )
            invoice = await self.service.create_invoice(invoice_create)
            
            # Verify calculations
            expected_subtotal = (2 * 100.00) + (1 * 50.00)  # 250.00
            assert invoice.subtotal == expected_subtotal, f"Subtotal incorrect: {invoice.subtotal} != {expected_subtotal}"
            assert invoice.total_amount > 0, "Total amount not calculated"
            assert invoice.amount_due == invoice.total_amount, "Amount due should equal total"
            
            self.record_result("Invoice Calculations", True, "Calculations correct")
        except Exception as e:
            self.record_result("Invoice Calculations", False, str(e))
    
    async def test_statistics(self):
        """Test finance statistics"""
        try:
            stats = await self.service.get_stats()
            assert stats is not None, "Stats not returned"
            assert hasattr(stats, 'total_accounts'), "Missing total_accounts"
            assert hasattr(stats, 'total_transactions'), "Missing total_transactions"
            assert hasattr(stats, 'total_invoices'), "Missing total_invoices"
            assert hasattr(stats, 'total_revenue'), "Missing total_revenue"
            assert hasattr(stats, 'total_expenses'), "Missing total_expenses"
            assert hasattr(stats, 'net_income'), "Missing net_income"
            
            self.record_result("Statistics", True, f"Stats: {stats.total_accounts} accounts, {stats.total_transactions} transactions, {stats.total_invoices} invoices")
        except Exception as e:
            self.record_result("Statistics", False, str(e))
    
    async def test_filtering(self):
        """Test filtering capabilities"""
        try:
            # Create multiple accounts of different types
            for i, acc_type in enumerate([AccountType.ASSET, AccountType.LIABILITY, AccountType.REVENUE]):
                account_create = AccountCreate(
                    name=f"Test {acc_type.value} {i}",
                    account_number=f"TST-{i}",
                    account_type=acc_type,
                    account_subtype=AccountSubType.CASH if acc_type == AccountType.ASSET else AccountSubType.ACCOUNTS_PAYABLE,
                    currency="USD"
                )
                await self.service.create_account(account_create)
            
            # Filter by type
            asset_accounts = await self.service.list_accounts(account_type=AccountType.ASSET)
            assert len(asset_accounts) >= 1, "No asset accounts found"
            
            # Filter by active status
            active_accounts = await self.service.list_accounts(is_active=True)
            assert len(active_accounts) >= 1, "No active accounts found"
            
            self.record_result("Filtering", True, "Filtering works correctly")
        except Exception as e:
            self.record_result("Filtering", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("🧪 FINANCE MODULE TEST SUITE")
        print("="*80 + "\n")
        
        await self.setup()
        
        # Run tests
        await self.test_account_crud()
        await self.test_transaction_crud()
        await self.test_invoice_crud()
        await self.test_account_balance_updates()
        await self.test_invoice_calculations()
        await self.test_statistics()
        await self.test_filtering()
        
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
    tester = TestFinanceModule()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

