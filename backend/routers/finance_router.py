"""
Finance Router for CPAS4
REST API endpoints for Finance Management
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
from services.finance_service import FinanceService
from models.finance_models import *
from database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/finance", tags=["Finance"])

async def get_service() -> FinanceService:
    db = Database()
    await db.connect()
    return FinanceService(db)

# Account endpoints
@router.post("/accounts", response_model=Account, status_code=201)
async def create_account(account: AccountCreate, svc: FinanceService = Depends(get_service)):
    return await svc.create_account(account)

@router.get("/accounts/{account_id}", response_model=Account)
async def get_account(account_id: str, svc: FinanceService = Depends(get_service)):
    acc = await svc.get_account(account_id)
    if not acc: raise HTTPException(404, "Account not found")
    return acc

@router.patch("/accounts/{account_id}", response_model=Account)
async def update_account(account_id: str, account_update: AccountUpdate, svc: FinanceService = Depends(get_service)):
    acc = await svc.update_account(account_id, account_update)
    if not acc: raise HTTPException(404, "Account not found")
    return acc

@router.delete("/accounts/{account_id}", status_code=204)
async def delete_account(account_id: str, svc: FinanceService = Depends(get_service)):
    if not await svc.delete_account(account_id): raise HTTPException(404, "Account not found")

@router.get("/accounts", response_model=List[Account])
async def list_accounts(account_type: Optional[AccountType] = None, is_active: Optional[bool] = None, limit: int = Query(100, le=1000), svc: FinanceService = Depends(get_service)):
    return await svc.list_accounts(account_type, is_active, limit)

# Transaction endpoints
@router.post("/transactions", response_model=Transaction, status_code=201)
async def create_transaction(transaction: TransactionCreate, svc: FinanceService = Depends(get_service)):
    return await svc.create_transaction(transaction)

@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str, svc: FinanceService = Depends(get_service)):
    txn = await svc.get_transaction(transaction_id)
    if not txn: raise HTTPException(404, "Transaction not found")
    return txn

@router.patch("/transactions/{transaction_id}", response_model=Transaction)
async def update_transaction(transaction_id: str, transaction_update: TransactionUpdate, svc: FinanceService = Depends(get_service)):
    txn = await svc.update_transaction(transaction_id, transaction_update)
    if not txn: raise HTTPException(404, "Transaction not found")
    return txn

@router.delete("/transactions/{transaction_id}", status_code=204)
async def delete_transaction(transaction_id: str, svc: FinanceService = Depends(get_service)):
    if not await svc.delete_transaction(transaction_id): raise HTTPException(404, "Transaction not found")

@router.get("/transactions", response_model=List[Transaction])
async def list_transactions(account_id: Optional[str] = None, transaction_type: Optional[TransactionType] = None, status: Optional[TransactionStatus] = None, limit: int = Query(100, le=1000), svc: FinanceService = Depends(get_service)):
    return await svc.list_transactions(account_id, transaction_type, status, limit)

# Invoice endpoints
@router.post("/invoices", response_model=Invoice, status_code=201)
async def create_invoice(invoice: InvoiceCreate, svc: FinanceService = Depends(get_service)):
    return await svc.create_invoice(invoice)

@router.get("/invoices/{invoice_id}", response_model=Invoice)
async def get_invoice(invoice_id: str, svc: FinanceService = Depends(get_service)):
    inv = await svc.get_invoice(invoice_id)
    if not inv: raise HTTPException(404, "Invoice not found")
    return inv

@router.patch("/invoices/{invoice_id}", response_model=Invoice)
async def update_invoice(invoice_id: str, invoice_update: InvoiceUpdate, svc: FinanceService = Depends(get_service)):
    inv = await svc.update_invoice(invoice_id, invoice_update)
    if not inv: raise HTTPException(404, "Invoice not found")
    return inv

@router.delete("/invoices/{invoice_id}", status_code=204)
async def delete_invoice(invoice_id: str, svc: FinanceService = Depends(get_service)):
    if not await svc.delete_invoice(invoice_id): raise HTTPException(404, "Invoice not found")

@router.get("/invoices", response_model=List[Invoice])
async def list_invoices(status: Optional[InvoiceStatus] = None, customer_id: Optional[str] = None, limit: int = Query(100, le=1000), svc: FinanceService = Depends(get_service)):
    return await svc.list_invoices(status, customer_id, limit)

# Statistics
@router.get("/stats", response_model=FinanceStats)
async def get_stats(svc: FinanceService = Depends(get_service)):
    return await svc.get_stats()

# Health check
@router.get("/health")
async def health_check():
    return {"status": "healthy", "module": "Finance", "features": ["Account Management", "Transaction Tracking", "Invoice Management", "Financial Reporting", "Budget Management", "Statistics"], "endpoints": 17, "database": "MongoDB/File Storage"}
