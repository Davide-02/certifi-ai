"""
Schema for driving license documents
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, List
from datetime import datetime
from .base import BaseDocumentSchema


class DrivingLicenseSchema(BaseDocumentSchema):
    """Schema for extracted driving license data"""
    
    document_type: Literal["driving_license"] = "driving_license"
    
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    full_name: Optional[str] = Field(None, description="Full name")
    
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    place_of_birth: Optional[str] = Field(None, description="Place of birth")
    
    license_number: Optional[str] = Field(None, description="License number")
    
    issue_date: Optional[datetime] = Field(None, description="Issue date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    
    issuing_authority: Optional[str] = Field(None, description="Issuing authority")
    
    categories: Optional[List[str]] = Field(None, description="License categories (A1, A2, B, C1, C, D1, D, etc.)")
    
    address: Optional[str] = Field(None, description="Address")
    
    @validator('date_of_birth', 'issue_date', 'expiry_date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            from dateutil import parser
            try:
                return parser.parse(v)
            except:
                return None
        return v
