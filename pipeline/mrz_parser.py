"""
MRZ (Machine Readable Zone) Parser for ID documents
MRZ is the most reliable source of truth for ID documents
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime


class MRZParser:
    """Parse MRZ from ID documents (TD1, TD2, TD3 formats)"""
    
    def __init__(self):
        # TD3 format (passport-like, 2 lines of 44 chars)
        self.td3_pattern = re.compile(
            r'([A-Z0-9<]{44})\n([A-Z0-9<]{44})'
        )
        
        # TD1 format (ID card, 3 lines of 30 chars)
        self.td1_pattern = re.compile(
            r'([A-Z0-9<]{30})\n([A-Z0-9<]{30})\n([A-Z0-9<]{30})'
        )
    
    def detect_mrz(self, text: str) -> Optional[str]:
        """
        Detect MRZ in text
        
        Args:
            text: Document text
            
        Returns:
            MRZ string if found, None otherwise
        """
        # Look for MRZ patterns
        lines = text.split('\n')
        mrz_lines = []
        
        for line in lines:
            # MRZ characteristics: long lines with <, A-Z, 0-9
            if re.match(r'^[A-Z0-9<]{25,}$', line.strip()):
                mrz_lines.append(line.strip())
        
        if len(mrz_lines) >= 2:
            return '\n'.join(mrz_lines)
        
        return None
    
    def parse_td3(self, mrz: str) -> Dict[str, Any]:
        """
        Parse TD3 format MRZ (passport-like, 2 lines)
        
        Format:
        Line 1: P<ITA... (document type, country, name)
        Line 2: ... (document number, date of birth, etc.)
        
        Args:
            mrz: MRZ string (2 lines)
            
        Returns:
            Parsed data dictionary
        """
        lines = mrz.strip().split('\n')
        if len(lines) < 2:
            return {}
        
        line1 = lines[0]
        line2 = lines[1]
        
        data = {}
        
        # Document type (first char)
        if len(line1) > 0:
            data['document_type_code'] = line1[0]
        
        # Country code (chars 2-4)
        if len(line1) >= 5:
            data['country_code'] = line1[2:5]
        
        # Name (chars 5-44, format: SURNAME<<FIRSTNAME<<MIDDLENAME)
        if len(line1) >= 44:
            name_part = line1[5:44].replace('<', ' ').strip()
            name_parts = [p for p in name_part.split('<<') if p]
            if len(name_parts) >= 1:
                data['surname'] = name_parts[0].strip()
            if len(name_parts) >= 2:
                data['given_names'] = ' '.join(name_parts[1:]).strip()
        
        # Document number (line 2, chars 0-9)
        if len(line2) >= 9:
            doc_num = line2[0:9].replace('<', '').strip()
            if doc_num:
                data['document_number'] = doc_num
        
        # Check digit (char 9)
        if len(line2) > 9:
            data['document_number_check'] = line2[9]
        
        # Nationality (chars 10-12)
        if len(line2) >= 13:
            data['nationality'] = line2[10:13]
        
        # Date of birth (chars 13-19, YYMMDD)
        if len(line2) >= 19:
            dob_str = line2[13:19]
            data['date_of_birth_raw'] = dob_str
            dob = self._parse_date(dob_str)
            if dob:
                data['date_of_birth'] = dob
        
        # Check digit (char 19)
        if len(line2) > 19:
            data['dob_check'] = line2[19]
        
        # Sex (char 20)
        if len(line2) > 20:
            sex = line2[20]
            if sex in ['M', 'F', '<']:
                data['sex'] = sex if sex != '<' else None
        
        # Date of expiry (chars 21-27, YYMMDD)
        if len(line2) >= 27:
            exp_str = line2[21:27]
            data['expiry_date_raw'] = exp_str
            exp = self._parse_date(exp_str)
            if exp:
                data['expiry_date'] = exp
        
        # Check digit (char 27)
        if len(line2) > 27:
            data['expiry_check'] = line2[27]
        
        # Optional data (chars 28-42)
        if len(line2) >= 42:
            optional = line2[28:42].replace('<', '').strip()
            if optional:
                data['optional_data'] = optional
        
        # Final check digit (char 42)
        if len(line2) > 42:
            data['final_check'] = line2[42]
        
        return data
    
    def parse_td1(self, mrz: str) -> Dict[str, Any]:
        """
        Parse TD1 format MRZ (ID card, 3 lines)
        
        Args:
            mrz: MRZ string (3 lines)
            
        Returns:
            Parsed data dictionary
        """
        lines = mrz.strip().split('\n')
        if len(lines) < 3:
            return {}
        
        line1 = lines[0]
        line2 = lines[1]
        line3 = lines[2]
        
        data = {}
        
        # Document type (first char of line 1)
        if len(line1) > 0:
            data['document_type_code'] = line1[0]
        
        # Country code (chars 2-4)
        if len(line1) >= 5:
            data['country_code'] = line1[2:5]
        
        # Document number (line 1, chars 5-14)
        if len(line1) >= 14:
            doc_num = line1[5:14].replace('<', '').strip()
            if doc_num:
                data['document_number'] = doc_num
        
        # Date of birth (line 2, chars 0-6, YYMMDD)
        if len(line2) >= 6:
            dob_str = line2[0:6]
            data['date_of_birth_raw'] = dob_str
            dob = self._parse_date(dob_str)
            if dob:
                data['date_of_birth'] = dob
        
        # Sex (line 2, char 7)
        if len(line2) > 7:
            sex = line2[7]
            if sex in ['M', 'F', '<']:
                data['sex'] = sex if sex != '<' else None
        
        # Date of expiry (line 2, chars 8-14, YYMMDD)
        if len(line2) >= 14:
            exp_str = line2[8:14]
            data['expiry_date_raw'] = exp_str
            exp = self._parse_date(exp_str)
            if exp:
                data['expiry_date'] = exp
        
        # Nationality (line 2, chars 15-17)
        if len(line2) >= 18:
            data['nationality'] = line2[15:18]
        
        # Name (line 3, format: SURNAME<<FIRSTNAME)
        if len(line3) >= 30:
            name_part = line3.replace('<', ' ').strip()
            name_parts = [p for p in name_part.split() if p]
            if len(name_parts) >= 1:
                data['surname'] = name_parts[0].strip()
            if len(name_parts) >= 2:
                data['given_names'] = ' '.join(name_parts[1:]).strip()
        
        return data
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Auto-detect and parse MRZ from text
        
        Args:
            text: Document text
            
        Returns:
            Parsed MRZ data with confidence
        """
        mrz = self.detect_mrz(text)
        if not mrz:
            return {
                'found': False,
                'data': {},
                'confidence': 0.0
            }
        
        lines = [l.strip() for l in mrz.split('\n') if l.strip()]
        
        # Try TD3 first (most common for passports)
        if len(lines) >= 2:
            td3_data = self.parse_td3(mrz)
            if td3_data and ('surname' in td3_data or 'given_names' in td3_data):
                return {
                    'found': True,
                    'format': 'TD3',
                    'data': td3_data,
                    'confidence': 0.95,  # MRZ is highly reliable
                    'raw_mrz': mrz
                }
        
        # Try TD1 (ID cards)
        if len(lines) >= 3:
            td1_data = self.parse_td1(mrz)
            if td1_data and ('surname' in td1_data or 'given_names' in td1_data):
                return {
                    'found': True,
                    'format': 'TD1',
                    'data': td1_data,
                    'confidence': 0.95,
                    'raw_mrz': mrz
                }
        
        # Fallback: Try to extract names from any MRZ-like line
        # Some documents have names in a separate line
        data = {}
        for line in lines:
            # Look for name pattern: SURNAME<<FIRSTNAME or SURNAME FIRSTNAME
            if '<<' in line and len(line) > 10:
                # Format: SURNAME<<FIRSTNAME
                name_parts = [p.strip() for p in line.split('<<') if p.strip()]
                if len(name_parts) >= 2:
                    data['surname'] = name_parts[0]
                    data['given_names'] = ' '.join(name_parts[1:])
                    break
            elif re.match(r'^[A-Z]+\s+[A-Z]+$', line) and len(line.split()) == 2:
                # Format: SURNAME FIRSTNAME (space separated)
                parts = line.split()
                if len(parts) == 2:
                    data['surname'] = parts[0]
                    data['given_names'] = parts[1]
                    break
        
        # Try to extract dates and numbers from other lines
        for line in lines:
            # Date of birth pattern: YYMMDD
            dob_match = re.search(r'(\d{6})', line)
            if dob_match and not data.get('date_of_birth'):
                dob = self._parse_date(dob_match.group(1))
                if dob:
                    data['date_of_birth'] = dob
            
            # Document number (alphanumeric, 6-15 chars)
            doc_match = re.search(r'([A-Z0-9]{6,15})', line)
            if doc_match and not data.get('document_number'):
                doc_num = doc_match.group(1)
                if not doc_num.isdigit():  # Likely a document number
                    data['document_number'] = doc_num
        
        if data:
            return {
                'found': True,
                'format': 'CUSTOM',
                'data': data,
                'confidence': 0.8,  # Lower confidence for non-standard format
                'raw_mrz': mrz
            }
        
        return {
            'found': True,
            'format': 'UNKNOWN',
            'data': {},
            'confidence': 0.5,
            'raw_mrz': mrz
        }
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse MRZ date format (YYMMDD)
        
        Args:
            date_str: Date string in YYMMDD format
            
        Returns:
            datetime object or None
        """
        if not date_str or len(date_str) != 6:
            return None
        
        try:
            year = int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            # Handle century (assume 1900-2099)
            if year < 50:
                year += 2000
            else:
                year += 1900
            
            return datetime(year, month, day)
        except (ValueError, TypeError):
            return None
