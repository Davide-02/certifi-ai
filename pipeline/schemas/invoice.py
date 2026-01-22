"""
Schema for invoice documents
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from decimal import Decimal
from .base import BaseDocumentSchema


class InvoiceSchema(BaseDocumentSchema):
    """Schema for extracted invoice data"""
    
    document_type: Literal["invoice"] = "invoice"
    
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[datetime] = Field(None, description="Invoice date")
    
    seller_name: Optional[str] = Field(None, description="Seller/issuer name")
    seller_vat: Optional[str] = Field(None, description="Seller VAT number")
    seller_address: Optional[str] = Field(None, description="Seller address")
    
    buyer_name: Optional[str] = Field(None, description="Buyer/client name")
    buyer_vat: Optional[str] = Field(None, description="Buyer VAT number")
    
    total_amount: Optional[Decimal] = Field(None, description="Total amount")
    vat_amount: Optional[Decimal] = Field(None, description="VAT amount")
    vat_rate: Optional[float] = Field(None, description="VAT rate percentage")
    net_amount: Optional[Decimal] = Field(None, description="Net amount")
    
    currency: Optional[str] = Field(default="EUR", description="Currency code")
    
    @validator('invoice_date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            # Try to parse various date formats
            from dateutil import parser
            try:
                return parser.parse(v)
            except:
                return None
        return v
