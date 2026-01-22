"""
Schema for diploma/degree documents
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from .base import BaseDocumentSchema


class DiplomaSchema(BaseDocumentSchema):
    """Schema for extracted diploma/degree data"""
    
    document_type: Literal["diploma"] = "diploma"
    
    student_name: Optional[str] = Field(None, description="Student full name")
    student_id: Optional[str] = Field(None, description="Student ID/registration number")
    
    degree_type: Optional[str] = Field(None, description="Type of degree (e.g., 'Laurea Triennale')")
    field_of_study: Optional[str] = Field(None, description="Field of study/major")
    
    university_name: Optional[str] = Field(None, description="University name")
    university_location: Optional[str] = Field(None, description="University location")
    
    graduation_date: Optional[datetime] = Field(None, description="Graduation date")
    final_grade: Optional[str] = Field(None, description="Final grade/mark")
    
    cfu_total: Optional[int] = Field(None, description="Total CFU credits")
    cfu_earned: Optional[int] = Field(None, description="CFU credits earned")
    
    thesis_title: Optional[str] = Field(None, description="Thesis title")
    
    @validator('graduation_date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            from dateutil import parser
            try:
                return parser.parse(v)
            except:
                return None
        return v
