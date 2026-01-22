"""
Schema for ID documents
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from .base import BaseDocumentSchema


class IDDocumentSchema(BaseDocumentSchema):
    """Schema for extracted ID document data"""
    
    document_type: Literal["id"] = "id"
    
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    full_name: Optional[str] = Field(None, description="Full name")
    
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    place_of_birth: Optional[str] = Field(None, description="Place of birth")
    
    nationality: Optional[str] = Field(None, description="Nationality")
    
    document_number: Optional[str] = Field(None, description="Document number")
    tax_code: Optional[str] = Field(None, description="Tax code (codice fiscale)")
    
    address: Optional[str] = Field(None, description="Address")
    city: Optional[str] = Field(None, description="City")
    postal_code: Optional[str] = Field(None, description="Postal code")
    
    issue_date: Optional[datetime] = Field(None, description="Document issue date")
    expiry_date: Optional[datetime] = Field(None, description="Document expiry date")
    
    issuing_authority: Optional[str] = Field(None, description="Issuing authority")
    
    @validator('date_of_birth', 'issue_date', 'expiry_date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            from dateutil import parser
            try:
                return parser.parse(v)
            except:
                return None
        return v
    
    @validator('tax_code')
    def validate_tax_code(cls, v):
        if v:
            # Basic Italian tax code validation (16 chars, alphanumeric)
            if len(v) == 16 and v.isalnum():
                return v.upper()
        return v
