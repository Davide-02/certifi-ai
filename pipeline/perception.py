"""
PERCEPTION LAYER

Extracts raw signals from documents without interpretation or classification.

Responsibilities:
- Extract raw text (OCR, PDF parsing)
- Analyze layout structure (titles, sections, tables)
- Detect language
- Extract visual signals (signatures, stamps, quality)

NO classification, NO interpretation - just raw data extraction.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .ocr import TextExtractor
from .layout_analyzer import LayoutAnalyzer
from .vision_analyzer import VisionAnalyzer
from .table_extractor import TableExtractor
from .document_parser import DocumentParser


class Language(str, Enum):
    """Detected language"""
    ENGLISH = "en"
    ITALIAN = "it"
    ARABIC = "ar"
    MULTILINGUAL = "multilingual"
    UNKNOWN = "unknown"


@dataclass
class PerceptionResult:
    """Raw perception data - no interpretation"""
    
    # Text extraction
    raw_text: str
    text_length: int
    text_preview: str  # First 500 chars
    
    # Layout signals
    has_titles: bool
    has_sections: bool
    has_tables: bool
    has_numbered_fields: bool
    has_form_fields: bool
    
    # Visual signals
    has_signatures: bool
    has_stamps: bool
    image_quality: float  # 0.0 - 1.0
    orientation: str  # "portrait", "landscape", "unknown"
    
    # Table extraction
    extracted_tables: List[Dict[str, Any]]
    
    # Language
    detected_language: Language
    
    # Metadata
    page_count: int
    file_type: str
    
    # Explainability
    extraction_methods: List[str]  # ["pdfplumber", "tesseract", "camelot", etc.]
    confidence_signals: Dict[str, float]  # Confidence for each extraction method


class PerceptionLayer:
    """
    PERCEPTION LAYER - Raw signal extraction
    
    Extracts all observable signals from a document without interpretation.
    This layer is purely descriptive - it does NOT classify or interpret.
    """
    
    def __init__(self):
        """Initialize perception layer components"""
        self.text_extractor = TextExtractor()
        self.layout_analyzer = LayoutAnalyzer()
        self.vision_analyzer = VisionAnalyzer()
        self.table_extractor = TableExtractor()
        self.document_parser = DocumentParser()
    
    def perceive(self, file_path: Union[str, Path]) -> PerceptionResult:
        """
        Extract all raw signals from document
        
        Args:
            file_path: Path to document file
            
        Returns:
            PerceptionResult with all raw signals
        """
        file_path = Path(file_path)
        
        # 1. Extract raw text
        raw_text = self.text_extractor.extract(str(file_path))
        text_length = len(raw_text)
        text_preview = raw_text[:500] if text_length > 500 else raw_text
        
        # 2. Analyze layout structure
        layout_result = self.layout_analyzer.analyze(raw_text, str(file_path))
        
        # 3. Analyze visual signals
        vision_result = self.vision_analyzer.analyze(str(file_path))
        
        # 4. Extract tables (IMPROVED: Try Camelot first, fallback to PyMuPDF blocks)
        extracted_tables = []
        if self.table_extractor.available:
            try:
                tables = self.table_extractor.extract_tables(str(file_path))
                extracted_tables = [
                    {
                        'type': t.get('type', 'unknown'),
                        'page': t.get('page', 0),
                        'shape': t.get('shape', (0, 0)),
                        'data': t.get('data', [])[:5]  # Limit size
                    }
                    for t in tables
                ]
            except Exception:
                pass
        
        # FALLBACK: If no tables found with Camelot, try PyMuPDF blocks
        if not extracted_tables:
            try:
                block_tables = self.table_extractor.extract_tables_from_pdf_blocks(str(file_path))
                extracted_tables = [
                    {
                        'type': t.get('type', 'text_block'),
                        'page': t.get('page', 0),
                        'shape': t.get('shape', (0, 0)),
                        'data': t.get('data', [])[:5]  # Limit size
                    }
                    for t in block_tables
                ]
            except Exception:
                pass
        
        # 5. Detect language
        detected_language = self._detect_language(raw_text)
        
        # 6. Get file metadata
        file_type = file_path.suffix.lower()
        page_count = self._get_page_count(str(file_path))
        
        # 7. Build extraction methods list
        extraction_methods = []
        if raw_text:
            extraction_methods.append("text_extraction")
        if layout_result.get('has_mrz'):
            extraction_methods.append("mrz_detection")
        if extracted_tables:
            extraction_methods.append("table_extraction")
        if vision_result.get('has_signatures'):
            extraction_methods.append("signature_detection")
        
        # 8. Build confidence signals
        confidence_signals = {
            'text_extraction': 1.0 if raw_text else 0.0,
            'layout_analysis': layout_result.get('confidence', 0.0),
            'vision_analysis': vision_result.get('confidence', 0.0),
            'table_extraction': 1.0 if extracted_tables else 0.0,
        }
        
        return PerceptionResult(
            raw_text=raw_text,
            text_length=text_length,
            text_preview=text_preview,
            has_titles=layout_result.get('has_titles', False),
            has_sections=layout_result.get('has_sections', False),
            has_tables=len(extracted_tables) > 0,
            has_numbered_fields=layout_result.get('has_numbered_fields', False),
            has_form_fields=layout_result.get('has_form_fields', False),
            has_signatures=vision_result.get('has_signatures', False),
            has_stamps=vision_result.get('has_stamps', False),
            image_quality=vision_result.get('quality_score', 0.0),
            orientation=vision_result.get('orientation', 'unknown'),
            extracted_tables=extracted_tables,
            detected_language=detected_language,
            page_count=page_count,
            file_type=file_type,
            extraction_methods=extraction_methods,
            confidence_signals=confidence_signals
        )
    
    def _detect_language(self, text: str) -> Language:
        """Detect document language (simple heuristic)"""
        if not text or len(text.strip()) < 10:
            return Language.UNKNOWN
        
        text_lower = text.lower()
        
        # Simple heuristics
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        has_italian = any(word in text_lower for word in ['di', 'la', 'il', 'per', 'con', 'una', 'sono', 'essere'])
        has_english = any(word in text_lower for word in ['the', 'and', 'for', 'with', 'this', 'that', 'from'])
        
        if has_arabic and (has_english or has_italian):
            return Language.MULTILINGUAL
        elif has_arabic:
            return Language.ARABIC
        elif has_italian and not has_english:
            return Language.ITALIAN
        elif has_english:
            return Language.ENGLISH
        else:
            return Language.UNKNOWN
    
    def _get_page_count(self, file_path: str) -> int:
        """Get page count from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 1  # Default to 1 if can't determine
