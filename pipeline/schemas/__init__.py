"""
Pydantic schemas for document data structures
"""

from .invoice import InvoiceSchema
from .diploma import DiplomaSchema
from .id_document import IDDocumentSchema
from .driving_license import DrivingLicenseSchema
from .base import BaseDocumentSchema

__all__ = [
    'InvoiceSchema',
    'DiplomaSchema',
    'IDDocumentSchema',
    'DrivingLicenseSchema',
    'BaseDocumentSchema',
]
