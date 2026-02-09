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
        
        IMPROVED: Also uses PyMuPDF for table-like blocks as fallback
        
        Args:
            file_path: Path to PDF file
            pages: Page numbers (e.g., '1', '1-3', 'all'). Default: all pages
            flavor: 'lattice' (bordered tables) or 'stream' (borderless tables)
            
        Returns:
            List of extracted tables as dictionaries
        """
        extracted_tables = []
        
        # Try Camelot first (best for structured tables)
        if self.available:
            try:
                # Extract tables
                tables = camelot.read_pdf(file_path, pages=pages or 'all', flavor=flavor)
                
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
        
        # FALLBACK: Use PyMuPDF to extract table-like blocks
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")
                for block in blocks:
                    bbox_text = block[4].strip() if len(block) > 4 else ""
                    # Check if block looks like a table (has tabs, pipes, or multiple rows)
                    if bbox_text and ("\t" in bbox_text or "|" in bbox_text or 
                                     len(bbox_text.split('\n')) > 2):
                        # Try to parse as table
                        lines = bbox_text.split('\n')
                        if len(lines) >= 2:
                            # Convert to dict format
                            table_data = []
                            for line in lines:
                                if line.strip():
                                    # Split by tab or pipe
                                    if "\t" in line:
                                        row = [cell.strip() for cell in line.split("\t")]
                                    elif "|" in line:
                                        row = [cell.strip() for cell in line.split("|")]
                                    else:
                                        row = [line.strip()]
                                    
                                    if row:
                                        table_data.append(row)
                            
                            if table_data:
                                extracted_tables.append({
                                    'table_index': len(extracted_tables),
                                    'page': page_num + 1,
                                    'accuracy': 0.7,  # Lower confidence for block-based extraction
                                    'type': 'text_block',
                                    'data': table_data,
                                    'shape': (len(table_data), len(table_data[0]) if table_data else 0),
                                })
            doc.close()
        except Exception as e:
            print(f"⚠️  Error extracting tables with PyMuPDF blocks: {e}")
        
        return extracted_tables
    
    def extract_tables_from_pdf_blocks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract table-like content from PDF using PyMuPDF text blocks
        
        IMPROVED: Uses PyMuPDF to extract text blocks that may contain tables
        (identified by tabs or pipe separators)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of table-like text blocks as dictionaries
        """
        tables = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # Get text blocks
                blocks = page.get_text("blocks")
                for block in blocks:
                    bbox_text = block[4].strip() if len(block) > 4 else ""
                    # Check if block looks like a table (has tabs, pipes, or multiple rows)
                    if bbox_text and ("\t" in bbox_text or "|" in bbox_text or 
                                     len(bbox_text.split('\n')) > 2):
                        # Try to parse as table
                        lines = bbox_text.split('\n')
                        if len(lines) >= 2:
                            # Convert to dict format
                            table_data = []
                            for line in lines:
                                if line.strip():
                                    # Split by tab or pipe
                                    if "\t" in line:
                                        row = [cell.strip() for cell in line.split("\t")]
                                    elif "|" in line:
                                        row = [cell.strip() for cell in line.split("|")]
                                    else:
                                        row = [line.strip()]
                                    
                                    if row:
                                        table_data.append(row)
                            
                            if table_data:
                                tables.append({
                                    'table_index': len(tables),
                                    'page': page_num + 1,
                                    'accuracy': 0.7,  # Lower confidence for block-based extraction
                                    'type': 'text_block',
                                    'data': table_data,
                                    'shape': (len(table_data), len(table_data[0]) if table_data else 0),
                                })
            doc.close()
        except Exception as e:
            print(f"⚠️  Error extracting tables with PyMuPDF blocks: {e}")
        
        return tables
    
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
        """
        Parse compensation table into structured data
        
        IMPROVED: Looks for "Annual Contract Value" explicitly and extracts multi-currency amounts
        """
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
        
        # Detect if table has multi-currency columns (AED and USD)
        has_usd_column = False
        aed_column_idx = None
        usd_column_idx = None
        
        # Check header row for currency columns
        if len(df.columns) > 0:
            header_row = df.iloc[0] if len(df) > 0 else None
            if header_row is not None:
                for idx, cell in enumerate(header_row):
                    cell_str = str(cell).upper()
                    if 'AED' in cell_str or 'DIRHAM' in cell_str:
                        aed_column_idx = idx
                    elif 'USD' in cell_str or 'DOLLAR' in cell_str:
                        usd_column_idx = idx
                        has_usd_column = True
                        compensation['secondary_currency'] = 'USD'
        
        # Try to extract amounts from table
        for _, row in df.iterrows():
            row_text = ' '.join(str(cell).lower() for cell in row)
            
            # PRIORITY 1: Look for "Annual Contract Value" (highest priority)
            if 'annual contract value' in row_text or ('annual' in row_text and 'contract' in row_text and 'value' in row_text):
                amounts = self._extract_amounts_from_row(row)
                if amounts:
                    # Get the largest amount (likely the annual total)
                    largest_amount = max(amounts)
                    compensation['annual_total'] = largest_amount
                    
                    # If we have USD column, extract secondary amount
                    if has_usd_column and usd_column_idx is not None and usd_column_idx < len(row):
                        usd_cell = row.iloc[usd_column_idx] if hasattr(row, 'iloc') else row[usd_column_idx]
                        usd_amounts = self._extract_amounts_from_row([usd_cell])
                        if usd_amounts:
                            compensation['secondary_amounts']['annual_total'] = max(usd_amounts)
            
            # PRIORITY 2: Look for annual total (fallback)
            elif ('annual' in row_text or 'yearly' in row_text) and 'total' in row_text:
                amounts = self._extract_amounts_from_row(row)
                if amounts and not compensation['annual_total']:  # Only if not already set
                    compensation['annual_total'] = max(amounts)
                    
                    # Extract USD if available
                    if has_usd_column and usd_column_idx is not None and usd_column_idx < len(row):
                        usd_cell = row.iloc[usd_column_idx] if hasattr(row, 'iloc') else row[usd_column_idx]
                        usd_amounts = self._extract_amounts_from_row([usd_cell])
                        if usd_amounts:
                            compensation['secondary_amounts']['annual_total'] = max(usd_amounts)
            
            # PRIORITY 3: Look for monthly total
            elif 'monthly' in row_text and 'total' in row_text:
                amounts = self._extract_amounts_from_row(row)
                if amounts:
                    compensation['monthly_total'] = max(amounts)
                    
                    # Extract USD if available
                    if has_usd_column and usd_column_idx is not None and usd_column_idx < len(row):
                        usd_cell = row.iloc[usd_column_idx] if hasattr(row, 'iloc') else row[usd_column_idx]
                        usd_amounts = self._extract_amounts_from_row([usd_cell])
                        if usd_amounts:
                            compensation['secondary_amounts']['monthly_total'] = max(usd_amounts)
            
            # PRIORITY 4: Look for base fee
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
