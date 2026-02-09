"""
OCR and text extraction module
Handles PDF (text + scanned) and image files
"""

import os
from typing import Optional, Union, List
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
from .preprocessing import DocumentPreprocessor

# Try to import EasyOCR (optional, better OCR)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None


class TextExtractor:
    """Extracts text from various document formats"""
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
    
    def extract_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF (handles both text-based and scanned PDFs)
        
        IMPROVED: Uses PyMuPDF (fitz) for more reliable text extraction
        Also extracts table-like content from text blocks
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        # IMPROVED: Try PyMuPDF first for more reliable extraction
        try:
            doc = fitz.open(file_path)
            for page in doc:
                # Use "text" mode for better extraction
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            doc.close()
        except Exception as e:
            print(f"PyMuPDF failed: {e}")
        
        # Fallback to pdfplumber if PyMuPDF didn't work
        if not text_parts:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        
        # If still no text, it's likely a scanned PDF - use OCR
        if not text_parts or all(not t.strip() for t in text_parts):
            return self._ocr_pdf(file_path)
        
        return "\n\n".join(text_parts)
    
    def extract_tables_from_pdf_blocks(self, file_path: str) -> List[str]:
        """
        Extract table-like content from PDF using text blocks
        
        Uses PyMuPDF to extract text blocks that may contain tables
        (identified by tabs or pipe separators)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of table-like text blocks
        """
        tables = []
        try:
            doc = fitz.open(file_path)
            for page in doc:
                # Get text blocks
                blocks = page.get_text("blocks")
                for block in blocks:
                    bbox_text = block[4].strip() if len(block) > 4 else ""
                    # Check if block looks like a table (has tabs or pipes)
                    if bbox_text and ("\t" in bbox_text or "|" in bbox_text or 
                                     len(bbox_text.split('\n')) > 2):
                        tables.append(bbox_text)
            doc.close()
        except Exception as e:
            print(f"Table extraction from blocks failed: {e}")
        
        return tables
    
    def _ocr_pdf(self, file_path: str, use_easyocr: bool = False) -> str:
        """
        OCR a scanned PDF by converting pages to images
        
        Args:
            file_path: Path to PDF file
            use_easyocr: Use EasyOCR instead of Tesseract (better accuracy, slower)
            
        Returns:
            OCR'd text
        """
        text_parts = []
        
        # Use EasyOCR if available and requested (better for multi-language)
        if use_easyocr and EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR reader (English + Arabic for UAE documents)
                # Specify languages explicitly to avoid warning
                reader = easyocr.Reader(['en', 'ar'], gpu=False, verbose=False)
                
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_array = np.array(img)
                    
                    # EasyOCR
                    results = reader.readtext(img_array, detail=0)
                    page_text = '\n'.join(results)
                    if page_text.strip():
                        text_parts.append(page_text)
                
                doc.close()
                return "\n\n".join(text_parts)
            except Exception as e:
                print(f"EasyOCR failed: {e}, falling back to Tesseract")
        
        # Fallback to Tesseract
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Preprocess image
                img_array = np.array(img)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                preprocessed = self.preprocessor.preprocess_image_from_array(gray)
                
                if preprocessed is not None:
                    img = Image.fromarray(preprocessed)
                
                # OCR with Tesseract
                page_text = pytesseract.image_to_string(img, lang='ita+eng')
                if page_text.strip():
                    text_parts.append(page_text)
            
            doc.close()
        except Exception as e:
            print(f"OCR PDF failed: {e}")
        
        return "\n\n".join(text_parts)
    
    def extract_from_image(self, file_path: str, use_easyocr: bool = False) -> str:
        """
        Extract text from image file using OCR
        
        Args:
            file_path: Path to image file
            use_easyocr: Use EasyOCR instead of Tesseract (better accuracy, slower)
            
        Returns:
            Extracted text
        """
        # Use EasyOCR if available and requested
        if use_easyocr and EASYOCR_AVAILABLE:
            try:
                # Specify languages explicitly to avoid warning
                reader = easyocr.Reader(['en', 'ar'], gpu=False, verbose=False)
                results = reader.readtext(file_path, detail=0)
                return '\n'.join(results)
            except Exception as e:
                print(f"EasyOCR failed: {e}, falling back to Tesseract")
        
        # Fallback to Tesseract
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_image(file_path)
            
            if preprocessed is not None:
                img = Image.fromarray(preprocessed)
            else:
                img = Image.open(file_path)
            
            # OCR with Tesseract
            text = pytesseract.image_to_string(img, lang='ita+eng')
            return text
        except Exception as e:
            print(f"OCR image failed: {e}")
            return ""
    
    def extract(self, file_path: Union[str, Path]) -> str:
        """
        Main extraction method - auto-detects file type
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted and preprocessed text
        """
        file_path = str(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = self.extract_from_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            text = self.extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Preprocess extracted text
        return self.preprocessor.process(text)
