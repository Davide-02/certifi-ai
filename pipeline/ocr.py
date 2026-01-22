"""
OCR and text extraction module
Handles PDF (text + scanned) and image files
"""

import os
from typing import Optional, Union
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
from .preprocessing import DocumentPreprocessor


class TextExtractor:
    """Extracts text from various document formats"""
    
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
    
    def extract_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF (handles both text-based and scanned PDFs)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        # Try pdfplumber first (better for text-based PDFs)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        
        # If no text found, try PyMuPDF
        if not text_parts:
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text)
                doc.close()
            except Exception as e:
                print(f"PyMuPDF failed: {e}")
        
        # If still no text, it's likely a scanned PDF - use OCR
        if not text_parts or all(not t.strip() for t in text_parts):
            return self._ocr_pdf(file_path)
        
        return "\n\n".join(text_parts)
    
    def _ocr_pdf(self, file_path: str) -> str:
        """
        OCR a scanned PDF by converting pages to images
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            OCR'd text
        """
        text_parts = []
        
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
                
                # OCR
                page_text = pytesseract.image_to_string(img, lang='ita+eng')
                if page_text.strip():
                    text_parts.append(page_text)
            
            doc.close()
        except Exception as e:
            print(f"OCR PDF failed: {e}")
        
        return "\n\n".join(text_parts)
    
    def extract_from_image(self, file_path: str) -> str:
        """
        Extract text from image file using OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_image(file_path)
            
            if preprocessed is not None:
                img = Image.fromarray(preprocessed)
            else:
                img = Image.open(file_path)
            
            # OCR
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
