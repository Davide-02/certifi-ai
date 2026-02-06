"""
Document Parser - Intelligent document structure parsing using Unstructured

Detects document structure: headers, sections, tables, lists, etc.
"""

from typing import Dict, Any, List, Optional

# Try to import unstructured (optional)
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class DocumentParser:
    """
    Parses documents into structured elements using Unstructured
    
    Detects:
    - Titles and headers
    - Narrative text
    - Tables
    - Lists (numbered, bulleted)
    - Sections
    """
    
    def __init__(self):
        """Initialize document parser"""
        self.available = UNSTRUCTURED_AVAILABLE
        if not self.available:
            print("⚠️  unstructured not available. Install with: pip install unstructured[pdf]")
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse document into structured elements
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Structured document with elements categorized
        """
        if not self.available:
            return {
                'available': False,
                'elements': [],
                'sections': {},
                'tables': [],
                'lists': []
            }
        
        try:
            # Partition PDF into elements
            elements = partition_pdf(file_path)
            
            # Categorize elements
            structured = {
                'available': True,
                'elements': [],
                'sections': {},
                'tables': [],
                'lists': [],
                'headers': [],
                'narrative_text': []
            }
            
            current_section = None
            
            for element in elements:
                element_type = element.category if hasattr(element, 'category') else type(element).__name__
                element_text = str(element) if hasattr(element, '__str__') else ''
                
                element_data = {
                    'type': element_type,
                    'text': element_text[:500],  # Limit length
                    'metadata': element.metadata.__dict__ if hasattr(element, 'metadata') else {}
                }
                
                structured['elements'].append(element_data)
                
                # Categorize by type
                if element_type == 'Title' or 'Header' in element_type:
                    structured['headers'].append(element_data)
                    current_section = element_text[:50]  # Use header as section name
                    if current_section:
                        structured['sections'][current_section] = []
                
                elif element_type == 'NarrativeText':
                    structured['narrative_text'].append(element_data)
                    if current_section:
                        structured['sections'][current_section].append(element_data)
                
                elif element_type == 'Table':
                    structured['tables'].append(element_data)
                
                elif 'List' in element_type:
                    structured['lists'].append(element_data)
            
            return structured
            
        except Exception as e:
            print(f"⚠️  Error parsing document with Unstructured: {e}")
            return {
                'available': False,
                'error': str(e),
                'elements': []
            }
