"""
Adapter to bridge new ProductionPipeline with existing API interface

This allows gradual migration while maintaining backward compatibility.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

from .new_orchestrator import ProductionPipeline
from .orchestrator import DocumentPipeline  # Old pipeline


class PipelineAdapter:
    """
    Adapter that wraps ProductionPipeline to match DocumentPipeline interface
    
    This allows the API to use the new architecture transparently.
    """
    
    def __init__(self, use_new_pipeline: bool = True, use_llm: bool = False, llm_provider: str = "openai"):
        """
        Initialize adapter
        
        Args:
            use_new_pipeline: If True, use new ProductionPipeline; if False, use old DocumentPipeline
            use_llm: Whether to use LLM for semantic extraction
            llm_provider: "openai" or "anthropic"
        """
        self.use_new_pipeline = use_new_pipeline
        
        if use_new_pipeline:
            self.pipeline = ProductionPipeline(use_llm=use_llm, llm_provider=llm_provider)
        else:
            self.pipeline = DocumentPipeline(use_llm=use_llm, llm_provider=llm_provider)
    
    def process(
        self,
        file_path: Union[str, Path],
        document_type: Optional[str] = None,
        certification_profile: Optional[Any] = None,
        requested_tasks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process document - adapts between old and new pipeline interfaces
        
        Args:
            file_path: Path to document
            document_type: Optional (ignored in new pipeline - it determines type itself)
            certification_profile: Optional (not used in new pipeline)
            requested_tasks: List of tasks to perform
            
        Returns:
            Result dictionary compatible with existing API expectations
        """
        if self.use_new_pipeline:
            return self._process_with_new_pipeline(file_path, requested_tasks)
        else:
            return self.pipeline.process(
                file_path=file_path,
                document_type=document_type,
                certification_profile=certification_profile,
                requested_tasks=requested_tasks
            )
    
    def _process_with_new_pipeline(
        self,
        file_path: Union[str, Path],
        requested_tasks: Optional[list]
    ) -> Dict[str, Any]:
        """
        Process with new ProductionPipeline and adapt output to old format
        """
        # Process with new pipeline
        result = self.pipeline.process(file_path, requested_tasks)
        
        # Adapt output format to match old pipeline expectations
        adapted = {
            'success': result.get('success', False),
            'document_family': result.get('document_family', 'unknown'),
            'document_subtype': result.get('document_subtype'),
            'document_type': result.get('document_subtype') or result.get('document_family', 'unknown'),
            'confidence': result.get('confidence', 0.0),
            'data': None,  # Not used in new pipeline
            'validation': None,  # Not used in new pipeline
            'certification_ready': result.get('confidence', 0.0) > 0.7,  # Heuristic
            'human_review_required': result.get('confidence', 0.0) < 0.7,
            'metadata': {
                'perception': result.get('metadata', {}).get('perception', {}),
                'structure_analysis': result.get('metadata', {}).get('structure_analysis', {}),
                'extracted_facts': result.get('metadata', {}).get('extracted_facts', {}),
                'explanation': result.get('explanation', ''),
                'signals_used': result.get('signals_used', []),
                'alternatives_considered': result.get('alternatives_considered', []),
            },
            'errors': result.get('errors', []),
            # Direct mappings
            'holder': result.get('holder'),
            'claims': result.get('claims'),
            'compliance_score': result.get('compliance_score'),
            'anomalies': result.get('anomalies', []),
        }
        
        return adapted
