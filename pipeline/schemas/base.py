"""
Base schema for all document types
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class BaseDocumentSchema(BaseModel):
    """Base schema for all extracted documents"""
    
    document_type: str = Field(..., description="Type of document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    field_confidence: Dict[str, float] = Field(default_factory=dict, description="Confidence per field")
    extracted_at: datetime = Field(default_factory=datetime.now)
    raw_text: Optional[str] = Field(None, description="Original extracted text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Certification readiness
    certification_ready: bool = Field(default=False, description="Ready for on-chain certification")
    human_review_required: bool = Field(default=True, description="Requires human review")
    trusted_source: Optional[str] = Field(None, description="Source of truth (mrz, ocr, etc.)")
    missing_fields: list[str] = Field(default_factory=list, description="Missing required fields")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
