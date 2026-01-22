"""
Configuration file for CertiFi AI Pipeline
"""

import os
from typing import Optional


class Config:
    """Pipeline configuration"""
    
    # LLM Settings
    USE_LLM: bool = os.getenv("USE_LLM", "false").lower() == "true"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # OCR Settings
    TESSERACT_LANG: str = os.getenv("TESSERACT_LANG", "ita+eng")
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD")  # Custom path if needed
    
    # Processing Settings
    MIN_TEXT_LENGTH: int = int(os.getenv("MIN_TEXT_LENGTH", "10"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.5"))
    
    # Output Settings
    SAVE_RAW_TEXT: bool = os.getenv("SAVE_RAW_TEXT", "true").lower() == "true"
    SAVE_METADATA: bool = os.getenv("SAVE_METADATA", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if cls.USE_LLM:
            if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
                print("⚠️  Warning: USE_LLM=True but OPENAI_API_KEY not set")
                return False
            elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
                print("⚠️  Warning: USE_LLM=True but ANTHROPIC_API_KEY not set")
                return False
        return True
