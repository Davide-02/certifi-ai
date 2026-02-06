"""
Layout Analyzer - Analyzes document structure and layout

Part of the pipeline: OCR → Layout → Vision → LLM → Normalizer → JSON schema
"""

from typing import Dict, Any, List, Optional
import re


class LayoutAnalyzer:
    """
    Analyzes document layout and structure
    
    Detects:
    - Document sections (header, body, footer)
    - Structured fields (numbered lists, tables, forms)
    - Layout patterns (MRZ, form fields, etc.)
    """
    
    def analyze(self, text: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze document layout
        
        Args:
            text: Extracted text from document
            file_path: Optional file path for image-based analysis
            
        Returns:
            Layout analysis result
        """
        if not text:
            return {
                "has_structure": False,
                "sections": [],
                "structured_fields": [],
                "layout_type": "unstructured",
                "confidence": 0.0
            }
        
        lines = text.split('\n')
        
        # Detect sections
        sections = self._detect_sections(lines)
        
        # Detect structured fields
        structured_fields = self._detect_structured_fields(text)
        
        # Determine layout type
        layout_type = self._determine_layout_type(text, structured_fields)
        
        # Calculate confidence
        confidence = self._calculate_layout_confidence(sections, structured_fields, layout_type)
        
        return {
            "has_structure": len(structured_fields) > 0 or len(sections) > 1,
            "sections": sections,
            "structured_fields": structured_fields,
            "layout_type": layout_type,
            "confidence": confidence
        }
    
    def _detect_sections(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect document sections (header, body, footer)"""
        sections = []
        
        # Simple heuristic: header is usually first 3-5 lines
        # Footer is usually last 2-3 lines
        if len(lines) > 5:
            header_lines = lines[:3]
            footer_lines = lines[-2:] if len(lines) > 5 else []
            
            if header_lines:
                sections.append({
                    "type": "header",
                    "lines": header_lines,
                    "start_line": 0,
                    "end_line": len(header_lines)
                })
            
            if footer_lines:
                sections.append({
                    "type": "footer",
                    "lines": footer_lines,
                    "start_line": len(lines) - len(footer_lines),
                    "end_line": len(lines)
                })
        
        return sections
    
    def _detect_structured_fields(self, text: str) -> List[Dict[str, Any]]:
        """Detect structured fields (numbered lists, form fields, etc.)"""
        fields = []
        
        # Detect numbered fields (e.g., "1.", "2.", "4a.", "4b.")
        numbered_pattern = r'(\d+[a-z]?\.\s*[A-Z])'
        numbered_matches = re.finditer(numbered_pattern, text, re.IGNORECASE)
        for match in numbered_matches:
            fields.append({
                "type": "numbered_field",
                "pattern": match.group(0),
                "position": match.start()
            })
        
        # Detect form fields (e.g., "Name:", "Date:", "Address:")
        form_pattern = r'([A-Z][a-z]+)\s*:'
        form_matches = re.finditer(form_pattern, text)
        for match in form_matches:
            field_name = match.group(1)
            # Skip common false positives
            if field_name.lower() not in ['the', 'this', 'that', 'with', 'from']:
                fields.append({
                    "type": "form_field",
                    "name": field_name,
                    "position": match.start()
                })
        
        # Detect MRZ (Machine Readable Zone)
        mrz_pattern = r'[A-Z0-9<]{25,}'
        mrz_matches = re.finditer(mrz_pattern, text)
        for match in mrz_matches:
            fields.append({
                "type": "mrz",
                "pattern": match.group(0)[:30] + "...",
                "position": match.start()
            })
        
        return fields
    
    def _determine_layout_type(self, text: str, structured_fields: List[Dict[str, Any]]) -> str:
        """Determine layout type"""
        # Check for MRZ
        mrz_fields = [f for f in structured_fields if f.get("type") == "mrz"]
        if mrz_fields:
            return "mrz_document"
        
        # Check for numbered fields (forms, licenses)
        numbered_fields = [f for f in structured_fields if f.get("type") == "numbered_field"]
        if len(numbered_fields) >= 3:
            return "form_document"
        
        # Check for form fields
        form_fields = [f for f in structured_fields if f.get("type") == "form_field"]
        if len(form_fields) >= 5:
            return "form_document"
        
        # Check for table-like structure
        if re.search(r'\|\s*[A-Z]', text):  # Pipe-separated columns
            return "table_document"
        
        # Default
        return "unstructured"
    
    def _calculate_layout_confidence(
        self,
        sections: List[Dict[str, Any]],
        structured_fields: List[Dict[str, Any]],
        layout_type: str
    ) -> float:
        """Calculate confidence in layout analysis"""
        confidence = 0.0
        
        # Base confidence from structured fields
        if structured_fields:
            confidence += min(0.5, len(structured_fields) * 0.1)
        
        # Boost for specific layout types
        if layout_type == "mrz_document":
            confidence = max(confidence, 0.8)
        elif layout_type == "form_document":
            confidence = max(confidence, 0.6)
        
        # Boost for sections
        if len(sections) > 1:
            confidence = min(0.95, confidence + 0.2)
        
        return min(0.98, confidence)
