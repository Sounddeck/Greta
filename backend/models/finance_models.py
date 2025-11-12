"""
Finance Module Data Models for CPAS4
Complete financial management system with accounts, transactions, and invoices
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class AccountType(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class AccountSubType(str, Enum):
    # Assets
    CASH = "cash"
    BANK = "bank"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INVENTORY = "inventory"
    FIXED_ASSETS = "fixed_assets"
    # Liabilities
    ACCOUNTS_PAYABLE = "accounts_payable"
    CREDIT_CARD = "credit_card"
    LOAN = "loan"
    # Equity
    OWNERS_EQUITY = "owners_equity"
    RETAINED_EARNINGS = "retained_earnings"
    # Revenue
    SALES = "sales"
    SERVICE_REVENUE = "service_revenue"
    OTHER_INCOME = "other_income"
    # Expense
    COST_OF_GOODS = "cost_of_goods"
    OPERATING_EXPENSE = "operating_expense"
    PAYROLL = "payroll"
    UTILITIES = "utilities"
    RENT = "rent"
    MARKETING = "marketing"
    OTHER_EXPENSE = "other_expense"


class TransactionType(str, Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    CLEARED = "cleared"
    RECONCILED = "reconciled"
    VOID = "void"


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    VIEWED = "viewed"
    PARTIAL = "partial"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class PaymentMethod(str, Enum):
    CASH = "cash"
    CHECK = "check"
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    OTHER = "other"


# Account Models
class AccountBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    account_number: str = Field(..., min_length=1, max_length=50)
    account_type: AccountType
    account_subtype: AccountSubType
    description: Optional[str] = None
    currency: str = Field(default="USD", max_length=3)
    is_active: bool = True
    parent_account_id: Optional[str] = None
    tax_account: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AccountCreate(AccountBase):
    pass


class AccountUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    parent_account_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Account(AccountBase):
    account_id: str
    balance: float = 0.0
    created_at: datetime
    updated_at: datetime
    last_transaction_date: Optional[datetime] = None


# Transaction Models
class TransactionBase(BaseModel):
    transaction_type: TransactionType
    amount: float = Field(..., gt=0)
    from_account_id: str
    to_account_id: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    reference_number: Optional[str] = None
    transaction_date: datetime = Field(default_factory=datetime.utcnow)
    payment_method: Optional[PaymentMethod] = None
    attachments: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransactionCreate(TransactionBase):
    pass


class TransactionUpdate(BaseModel):
    transaction_type: Optional[TransactionType] = None
    amount: Optional[float] = Field(None, gt=0)
    category: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TransactionStatus] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Transaction(TransactionBase):
    transaction_id: str
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime
    updated_at: datetime
    created_by: str = "system"
    reconciled_at: Optional[datetime] = None


# Invoice Models
class InvoiceLineItem(BaseModel):
    description: str
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    tax_rate: float = Field(default=0.0, ge=0, le=100)
    discount: float = Field(default=0.0, ge=0)
    
    @property
    def subtotal(self) -> float:
        return self.quantity * self.unit_price
    
    @property
    def tax_amount(self) -> float:
        return (self.subtotal - self.discount) * (self.tax_rate / 100)
    
    @property
    def total(self) -> float:
        return self.subtotal - self.discount + self.tax_amount


class InvoiceBase(BaseModel):
    invoice_number: str = Field(..., min_length=1, max_length=50)
    customer_id: Optional[str] = None
    customer_name: str = Field(..., min_length=1, max_length=200)
    customer_email: Optional[str] = None
    customer_address: Optional[str] = None
    line_items: List[InvoiceLineItem] = Field(..., min_items=1)
    issue_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime
    payment_terms: str = Field(default="Net 30")
    notes: Optional[str] = None
    currency: str = Field(default="USD", max_length=3)
    tax_rate: float = Field(default=0.0, ge=0, le=100)
    discount: float = Field(default=0.0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('due_date')
    def due_date_must_be_after_issue_date(cls, v, values):
        if 'issue_date' in values and v < values['issue_date']:
            raise ValueError('due_date must be after issue_date')
        return v


class InvoiceCreate(InvoiceBase):
    pass


class InvoiceUpdate(BaseModel):
    customer_name: Optional[str] = Field(None, min_length=1, max_length=200)
    customer_email: Optional[str] = None
    customer_address: Optional[str] = None
    line_items: Optional[List[InvoiceLineItem]] = None
    due_date: Optional[datetime] = None
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[InvoiceStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class Invoice(InvoiceBase):
    invoice_id: str
    status: InvoiceStatus = InvoiceStatus.DRAFT
    subtotal: float = 0.0
    tax_amount: float = 0.0
    total_amount: float = 0.0
    amount_paid: float = 0.0
    amount_due: float = 0.0
    created_at: datetime
    updated_at: datetime
    sent_at: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    created_by: str = "system"


# Budget Models
class BudgetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    account_id: str
    period_start: datetime
    period_end: datetime
    budgeted_amount: float = Field(..., ge=0)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BudgetCreate(BudgetBase):
    pass


class BudgetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    budgeted_amount: Optional[float] = Field(None, ge=0)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Budget(BudgetBase):
    budget_id: str
    actual_amount: float = 0.0
    variance: float = 0.0
    variance_percentage: float = 0.0
    created_at: datetime
    updated_at: datetime


# Statistics Models
class FinanceStats(BaseModel):
    total_accounts: int
    active_accounts: int
    total_transactions: int
    total_invoices: int
    accounts_by_type: Dict[str, int]
    transactions_by_type: Dict[str, int]
    transactions_by_status: Dict[str, int]
    invoices_by_status: Dict[str, int]
    total_revenue: float
    total_expenses: float
    net_income: float
    total_assets: float
    total_liabilities: float
    total_equity: float
    accounts_receivable: float
    accounts_payable: float
    outstanding_invoices: int
    overdue_invoices: int
    total_invoice_amount: float
    total_paid_amount: float
    total_outstanding_amount: float


# Report Models
class FinancialReport(BaseModel):
    report_type: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    data: Dict[str, Any]
    summary: Dict[str, float]

