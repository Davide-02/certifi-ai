"""
Vision Analyzer - Image-based document analysis

Part of the pipeline: OCR → Layout → Vision → LLM → Normalizer → JSON schema

Uses computer vision techniques to analyze document images
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import cv2
import numpy as np


class VisionAnalyzer:
    """
    Analyzes documents using computer vision techniques
    
    Detects:
    - Document orientation
    - Image quality
    - Visual patterns (logos, stamps, signatures)
    - Text regions
    """
    
    def __init__(self):
        """Initialize vision analyzer"""
        pass
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze document image
        
        Args:
            file_path: Path to document image/PDF
            
        Returns:
            Vision analysis result
        """
        try:
            # Try to load image
            image = self._load_image(file_path)
            if image is None:
                return {
                    "image_available": False,
                    "orientation": "unknown",
                    "quality_score": 0.0,
                    "has_signatures": False,
                    "has_stamps": False,
                    "text_regions": [],
                    "confidence": 0.0
                }
            
            # Analyze image
            orientation = self._detect_orientation(image)
            quality_score = self._assess_quality(image)
            has_signatures = self._detect_signatures(image)
            has_stamps = self._detect_stamps(image)
            text_regions = self._detect_text_regions(image)
            
            # Calculate overall confidence
            confidence = self._calculate_vision_confidence(
                quality_score, has_signatures, has_stamps, len(text_regions)
            )
            
            return {
                "image_available": True,
                "orientation": orientation,
                "quality_score": quality_score,
                "has_signatures": has_signatures,
                "has_stamps": has_stamps,
                "text_regions": text_regions,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "image_available": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        """Load image from file"""
        try:
            # For PDFs, we'd need to convert to image first
            # For now, try direct image loading
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image = cv2.imread(file_path)
                return image
            # For PDFs, return None (would need pdf2image or similar)
            return None
        except Exception:
            return None
    
    def _detect_orientation(self, image: np.ndarray) -> str:
        """Detect document orientation"""
        height, width = image.shape[:2]
        
        # Simple heuristic: portrait vs landscape
        if height > width * 1.2:
            return "portrait"
        elif width > height * 1.2:
            return "landscape"
        else:
            return "square"
    
    def _assess_quality(self, image: np.ndarray) -> float:
        """Assess image quality (0.0 to 1.0)"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Normalize sharpness (typical range: 0-1000, good: >100)
            quality = min(1.0, sharpness / 500.0)
            
            # Check for blur
            if quality < 0.2:
                quality = 0.2  # Minimum quality
            
            return quality
            
        except Exception:
            return 0.5  # Default quality
    
    def _detect_signatures(self, image: np.ndarray) -> bool:
        """Detect if document has signatures"""
        # Simple heuristic: look for dark regions at bottom of document
        # In production, would use ML model or more sophisticated CV
        try:
            height, width = image.shape[:2]
            bottom_region = image[int(height * 0.8):, :]
            
            # Convert to grayscale
            if len(bottom_region.shape) == 3:
                gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = bottom_region
            
            # Look for dark regions (signatures are usually darker)
            dark_pixels = np.sum(gray < 100)
            total_pixels = gray.size
            
            # If more than 5% of bottom region is dark, likely has signature
            return (dark_pixels / total_pixels) > 0.05
            
        except Exception:
            return False
    
    def _detect_stamps(self, image: np.ndarray) -> bool:
        """Detect if document has stamps/seals"""
        # Simple heuristic: look for circular or rectangular colored regions
        # In production, would use ML model
        try:
            # Convert to HSV for color detection
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                return False
            
            # Look for red regions (common in stamps)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # If significant red regions found, likely has stamp
            red_ratio = np.sum(mask > 0) / mask.size
            return red_ratio > 0.01
            
        except Exception:
            return False
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in image"""
        regions = []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use edge detection to find text regions
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (potential text regions)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (text regions are usually rectangular)
            for contour in contours[:10]:  # Limit to top 10
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size (reasonable text region)
                if 100 < area < (image.shape[0] * image.shape[1] * 0.3):
                    regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area)
                    })
            
        except Exception:
            pass
        
        return regions
    
    def _calculate_vision_confidence(
        self,
        quality_score: float,
        has_signatures: bool,
        has_stamps: bool,
        text_regions_count: int
    ) -> float:
        """Calculate overall vision analysis confidence"""
        confidence = quality_score * 0.5  # Base from quality
        
        # Boost for signatures/stamps (indicates official document)
        if has_signatures:
            confidence += 0.2
        if has_stamps:
            confidence += 0.1
        
        # Boost for text regions
        if text_regions_count > 0:
            confidence += min(0.2, text_regions_count * 0.05)
        
        return min(0.98, confidence)
