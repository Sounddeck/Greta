"""
Finance Service for CPAS4 - Real Database Integration
Complete financial management with accounts, transactions, and invoices
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Database
from models.finance_models import *

logger = logging.getLogger(__name__)

class FinanceService:
    def __init__(self, database: Database):
        self.db = database
        self.accounts_collection = "finance_accounts"
        self.transactions_collection = "finance_transactions"
        self.invoices_collection = "finance_invoices"
        self.budgets_collection = "finance_budgets"
        logger.info("✅ Finance Service initialized with real database")
    
    # Account CRUD
    async def create_account(self, account_create: AccountCreate) -> Account:
        account_id = f"acc_{uuid.uuid4().hex[:12]}"
        account = Account(account_id=account_id, **account_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        ad = account.model_dump()
        ad["created_at"], ad["updated_at"] = ad["created_at"].isoformat(), ad["updated_at"].isoformat()
        if ad.get("last_transaction_date"): ad["last_transaction_date"] = ad["last_transaction_date"].isoformat()
        await self.db.insert_one(self.accounts_collection, ad)
        logger.info(f"✅ Account created: {account_id}")
        return account
    
    async def get_account(self, account_id: str) -> Optional[Account]:
        ad = await self.db.find_one(self.accounts_collection, {"account_id": account_id})
        if not ad: return None
        for k in ["created_at", "updated_at", "last_transaction_date"]:
            if k in ad and ad[k] and isinstance(ad[k], str): ad[k] = datetime.fromisoformat(ad[k])
        return Account(**ad)
    
    async def update_account(self, account_id: str, account_update: AccountUpdate) -> Optional[Account]:
        ud = account_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        return await self.get_account(account_id) if await self.db.update_one(self.accounts_collection, {"account_id": account_id}, {"$set": ud}) else None
    
    async def delete_account(self, account_id: str) -> bool:
        return await self.db.delete_one(self.accounts_collection, {"account_id": account_id})
    
    async def list_accounts(self, account_type: Optional[AccountType] = None, is_active: Optional[bool] = None, limit: int = 100) -> List[Account]:
        f = {}
        if account_type: f["account_type"] = account_type
        if is_active is not None: f["is_active"] = is_active
        ads = await self.db.find_many(self.accounts_collection, f, limit=limit)
        accounts = []
        for ad in ads:
            for k in ["created_at", "updated_at", "last_transaction_date"]:
                if k in ad and ad[k] and isinstance(ad[k], str): ad[k] = datetime.fromisoformat(ad[k])
            accounts.append(Account(**ad))
        return accounts
    
    async def update_account_balance(self, account_id: str, amount: float, is_debit: bool = True):
        account = await self.get_account(account_id)
        if not account: return False
        new_balance = account.balance + amount if is_debit else account.balance - amount
        await self.db.update_one(self.accounts_collection, {"account_id": account_id}, {"$set": {"balance": new_balance, "last_transaction_date": datetime.utcnow().isoformat()}})
        return True
    
    # Transaction CRUD
    async def create_transaction(self, transaction_create: TransactionCreate) -> Transaction:
        transaction_id = f"txn_{uuid.uuid4().hex[:12]}"
        transaction = Transaction(transaction_id=transaction_id, **transaction_create.model_dump(), created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        td = transaction.model_dump()
        for k in ["created_at", "updated_at", "transaction_date", "reconciled_at"]:
            if td.get(k): td[k] = td[k].isoformat()
        await self.db.insert_one(self.transactions_collection, td)
        
        # Update account balances
        if transaction.transaction_type == TransactionType.INCOME:
            await self.update_account_balance(transaction.from_account_id, transaction.amount, True)
        elif transaction.transaction_type == TransactionType.EXPENSE:
            await self.update_account_balance(transaction.from_account_id, transaction.amount, False)
        elif transaction.transaction_type == TransactionType.TRANSFER and transaction.to_account_id:
            await self.update_account_balance(transaction.from_account_id, transaction.amount, False)
            await self.update_account_balance(transaction.to_account_id, transaction.amount, True)
        
        logger.info(f"✅ Transaction created: {transaction_id}")
        return transaction
    
    async def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        td = await self.db.find_one(self.transactions_collection, {"transaction_id": transaction_id})
        if not td: return None
        for k in ["created_at", "updated_at", "transaction_date", "reconciled_at"]:
            if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
        return Transaction(**td)
    
    async def update_transaction(self, transaction_id: str, transaction_update: TransactionUpdate) -> Optional[Transaction]:
        ud = transaction_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "status" in ud and ud["status"] == TransactionStatus.RECONCILED:
            ud["reconciled_at"] = datetime.utcnow().isoformat()
        return await self.get_transaction(transaction_id) if await self.db.update_one(self.transactions_collection, {"transaction_id": transaction_id}, {"$set": ud}) else None
    
    async def delete_transaction(self, transaction_id: str) -> bool:
        return await self.db.delete_one(self.transactions_collection, {"transaction_id": transaction_id})
    
    async def list_transactions(self, account_id: Optional[str] = None, transaction_type: Optional[TransactionType] = None, status: Optional[TransactionStatus] = None, limit: int = 100) -> List[Transaction]:
        f = {}
        if account_id: f["$or"] = [{"from_account_id": account_id}, {"to_account_id": account_id}]
        if transaction_type: f["transaction_type"] = transaction_type
        if status: f["status"] = status
        tds = await self.db.find_many(self.transactions_collection, f, limit=limit)
        transactions = []
        for td in tds:
            for k in ["created_at", "updated_at", "transaction_date", "reconciled_at"]:
                if k in td and td[k] and isinstance(td[k], str): td[k] = datetime.fromisoformat(td[k])
            transactions.append(Transaction(**td))
        return transactions
    
    # Invoice CRUD
    async def create_invoice(self, invoice_create: InvoiceCreate) -> Invoice:
        invoice_id = f"inv_{uuid.uuid4().hex[:12]}"
        
        # Calculate totals
        subtotal = sum(item.subtotal for item in invoice_create.line_items)
        discount = invoice_create.discount
        tax_amount = sum(item.tax_amount for item in invoice_create.line_items)
        total_amount = subtotal - discount + tax_amount
        
        invoice = Invoice(invoice_id=invoice_id, **invoice_create.model_dump(), subtotal=subtotal, tax_amount=tax_amount, total_amount=total_amount, amount_due=total_amount, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        invd = invoice.model_dump()
        for k in ["created_at", "updated_at", "issue_date", "due_date", "sent_at", "paid_at"]:
            if invd.get(k): invd[k] = invd[k].isoformat()
        await self.db.insert_one(self.invoices_collection, invd)
        logger.info(f"✅ Invoice created: {invoice_id}")
        return invoice
    
    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        invd = await self.db.find_one(self.invoices_collection, {"invoice_id": invoice_id})
        if not invd: return None
        for k in ["created_at", "updated_at", "issue_date", "due_date", "sent_at", "paid_at"]:
            if k in invd and invd[k] and isinstance(invd[k], str): invd[k] = datetime.fromisoformat(invd[k])
        return Invoice(**invd)
    
    async def update_invoice(self, invoice_id: str, invoice_update: InvoiceUpdate) -> Optional[Invoice]:
        ud = invoice_update.model_dump(exclude_unset=True)
        ud["updated_at"] = datetime.utcnow().isoformat()
        if "status" in ud:
            if ud["status"] == InvoiceStatus.SENT:
                invoice = await self.get_invoice(invoice_id)
                if invoice and not invoice.sent_at: ud["sent_at"] = datetime.utcnow().isoformat()
            elif ud["status"] == InvoiceStatus.PAID:
                invoice = await self.get_invoice(invoice_id)
                if invoice and not invoice.paid_at:
                    ud["paid_at"] = datetime.utcnow().isoformat()
                    ud["amount_paid"] = invoice.total_amount
                    ud["amount_due"] = 0.0
        return await self.get_invoice(invoice_id) if await self.db.update_one(self.invoices_collection, {"invoice_id": invoice_id}, {"$set": ud}) else None
    
    async def delete_invoice(self, invoice_id: str) -> bool:
        return await self.db.delete_one(self.invoices_collection, {"invoice_id": invoice_id})
    
    async def list_invoices(self, status: Optional[InvoiceStatus] = None, customer_id: Optional[str] = None, limit: int = 100) -> List[Invoice]:
        f = {}
        if status: f["status"] = status
        if customer_id: f["customer_id"] = customer_id
        invds = await self.db.find_many(self.invoices_collection, f, limit=limit)
        invoices = []
        for invd in invds:
            for k in ["created_at", "updated_at", "issue_date", "due_date", "sent_at", "paid_at"]:
                if k in invd and invd[k] and isinstance(invd[k], str): invd[k] = datetime.fromisoformat(invd[k])
            invoices.append(Invoice(**invd))
        return invoices
    
    # Statistics
    async def get_stats(self) -> FinanceStats:
        all_accounts = await self.list_accounts(limit=10000)
        all_transactions = await self.list_transactions(limit=10000)
        all_invoices = await self.list_invoices(limit=10000)
        
        accounts_by_type = {t.value: sum(1 for a in all_accounts if a.account_type == t) for t in AccountType}
        transactions_by_type = {t.value: sum(1 for tx in all_transactions if tx.transaction_type == t) for t in TransactionType}
        transactions_by_status = {s.value: sum(1 for tx in all_transactions if tx.status == s) for s in TransactionStatus}
        invoices_by_status = {s.value: sum(1 for inv in all_invoices if inv.status == s) for s in InvoiceStatus}
        
        total_revenue = sum(tx.amount for tx in all_transactions if tx.transaction_type == TransactionType.INCOME)
        total_expenses = sum(tx.amount for tx in all_transactions if tx.transaction_type == TransactionType.EXPENSE)
        net_income = total_revenue - total_expenses
        
        total_assets = sum(a.balance for a in all_accounts if a.account_type == AccountType.ASSET)
        total_liabilities = sum(a.balance for a in all_accounts if a.account_type == AccountType.LIABILITY)
        total_equity = sum(a.balance for a in all_accounts if a.account_type == AccountType.EQUITY)
        
        accounts_receivable = sum(a.balance for a in all_accounts if a.account_subtype == AccountSubType.ACCOUNTS_RECEIVABLE)
        accounts_payable = sum(a.balance for a in all_accounts if a.account_subtype == AccountSubType.ACCOUNTS_PAYABLE)
        
        outstanding_invoices = sum(1 for inv in all_invoices if inv.status in [InvoiceStatus.SENT, InvoiceStatus.VIEWED, InvoiceStatus.PARTIAL, InvoiceStatus.OVERDUE])
        overdue_invoices = sum(1 for inv in all_invoices if inv.status == InvoiceStatus.OVERDUE)
        total_invoice_amount = sum(inv.total_amount for inv in all_invoices)
        total_paid_amount = sum(inv.amount_paid for inv in all_invoices)
        total_outstanding_amount = sum(inv.amount_due for inv in all_invoices)
        
        return FinanceStats(
            total_accounts=len(all_accounts), active_accounts=sum(1 for a in all_accounts if a.is_active),
            total_transactions=len(all_transactions), total_invoices=len(all_invoices),
            accounts_by_type=accounts_by_type, transactions_by_type=transactions_by_type,
            transactions_by_status=transactions_by_status, invoices_by_status=invoices_by_status,
            total_revenue=round(total_revenue, 2), total_expenses=round(total_expenses, 2), net_income=round(net_income, 2),
            total_assets=round(total_assets, 2), total_liabilities=round(total_liabilities, 2), total_equity=round(total_equity, 2),
            accounts_receivable=round(accounts_receivable, 2), accounts_payable=round(accounts_payable, 2),
            outstanding_invoices=outstanding_invoices, overdue_invoices=overdue_invoices,
            total_invoice_amount=round(total_invoice_amount, 2), total_paid_amount=round(total_paid_amount, 2),
            total_outstanding_amount=round(total_outstanding_amount, 2)
        )
