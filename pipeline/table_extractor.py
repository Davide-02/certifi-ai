"""
Table Extractor - Professional table extraction using Camelot

Handles complex tables with merged cells, borders, and complex layouts
"""

from typing import Dict, Any, List, Optional

# Try to import pandas (required by camelot)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Try to import camelot (optional)
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    camelot = None


class TableExtractor:
    """
    Extracts tables from PDFs using Camelot
    
    Camelot excels at:
    - Tables with borders
    - Merged cells
    - Complex layouts
    - Multi-page tables
    """
    
    def __init__(self):
        """Initialize table extractor"""
        self.available = CAMELOT_AVAILABLE and PANDAS_AVAILABLE
        if not CAMELOT_AVAILABLE:
            print("⚠️  camelot-py not available. Install with: pip install camelot-py[cv]")
        if not PANDAS_AVAILABLE:
            print("⚠️  pandas not available. Install with: pip install pandas")
    
    def extract_tables(
        self,
        file_path: str,
        pages: Optional[str] = None,
        flavor: str = 'lattice'  # 'lattice' for bordered tables, 'stream' for borderless
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF
        
        Args:
            file_path: Path to PDF file
            pages: Page numbers (e.g., '1', '1-3', 'all'). Default: all pages
            flavor: 'lattice' (bordered tables) or 'stream' (borderless tables)
            
        Returns:
            List of extracted tables as dictionaries
        """
        if not self.available:
            return []
        
        try:
            # Extract tables
            tables = camelot.read_pdf(file_path, pages=pages or 'all', flavor=flavor)
            
            extracted_tables = []
            for i, table in enumerate(tables):
                # Convert to dictionary
                df = table.df
                
                # Try to detect table type (compensation, parties, etc.)
                table_type = self._detect_table_type(df)
                
                # Convert DataFrame to dict (if pandas available)
                if PANDAS_AVAILABLE and df is not None:
                    try:
                        data_dict = df.to_dict('records')  # List of dicts
                        shape = df.shape  # (rows, cols)
                    except Exception:
                        data_dict = []
                        shape = (0, 0)
                else:
                    data_dict = []
                    shape = (0, 0)
                
                extracted_tables.append({
                    'table_index': i,
                    'page': table.page,
                    'accuracy': table.accuracy,
                    'type': table_type,
                    'data': data_dict,
                    'shape': shape,
                })
            
            return extracted_tables
            
        except Exception as e:
            print(f"⚠️  Error extracting tables with Camelot: {e}")
            return []
    
    def _detect_table_type(self, df) -> str:
        """Detect table type based on content"""
        if not PANDAS_AVAILABLE or df is None:
            return 'unknown'
        # Convert DataFrame to string for pattern matching
        table_text = df.to_string().lower()
        
        # Check for compensation table
        if any(term in table_text for term in ['compensation', 'salary', 'fee', 'amount', 'aed', 'usd', 'monthly', 'annual']):
            return 'compensation'
        
        # Check for parties table
        if any(term in table_text for term in ['party', 'client', 'contractor', 'name', 'address']):
            return 'parties'
        
        # Check for schedule/terms table
        if any(term in table_text for term in ['schedule', 'term', 'condition', 'clause']):
            return 'schedule'
        
        return 'unknown'
    
    def extract_compensation_table(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract compensation table specifically
        
        Returns structured compensation data
        """
        tables = self.extract_tables(file_path)
        
        # Find compensation table
        for table in tables:
            if table['type'] == 'compensation':
                # Get dataframe from camelot table if available
                if self.available and PANDAS_AVAILABLE:
                    try:
                        # Re-extract to get dataframe
                        all_tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
                        for t in all_tables:
                            if self._detect_table_type(t.df) == 'compensation':
                                return self._parse_compensation_table(t.df)
                    except Exception:
                        pass
                break
        
        return None
    
    def _parse_compensation_table(self, df) -> Dict[str, Any]:
        """Parse compensation table into structured data"""
        compensation = {
            'annual_total': None,
            'monthly_total': None,
            'base_fee': None,
            'allowances': {},
            'currency': 'AED',
            'secondary_currency': None,
            'secondary_amounts': {}
        }
        
        if not PANDAS_AVAILABLE or df is None:
            return compensation
        
        # Try to extract amounts from table
        # This is a basic implementation - can be enhanced based on actual table structure
        for _, row in df.iterrows():
            row_text = ' '.join(str(cell).lower() for cell in row)
            
            # Look for annual total
            if 'annual' in row_text or 'yearly' in row_text:
                # Extract amount from row
                amounts = self._extract_amounts_from_row(row)
                if amounts:
                    compensation['annual_total'] = max(amounts)
            
            # Look for monthly total
            elif 'monthly' in row_text and 'total' in row_text:
                amounts = self._extract_amounts_from_row(row)
                if amounts:
                    compensation['monthly_total'] = max(amounts)
            
            # Look for base fee
            elif 'base' in row_text or 'base fee' in row_text:
                amounts = self._extract_amounts_from_row(row)
                if amounts:
                    compensation['base_fee'] = max(amounts)
        
        return compensation
    
    def _extract_amounts_from_row(self, row) -> List[float]:
        """Extract numeric amounts from a table row"""
        import re
        amounts = []
        
        if not PANDAS_AVAILABLE:
            return amounts
        
        for cell in row:
            if pd.notna(cell):
                # Look for numbers with commas/dots
                matches = re.findall(r'[\d,]+\.?\d*', str(cell))
                for match in matches:
                    try:
                        amount = float(match.replace(',', ''))
                        if amount > 0:
                            amounts.append(amount)
                    except:
                        pass
        
        return amounts
