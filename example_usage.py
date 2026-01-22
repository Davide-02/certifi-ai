"""
Example usage of CertiFi AI Pipeline
"""

from pathlib import Path
from pipeline.orchestrator import DocumentPipeline


def example_basic():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize pipeline without LLM
    pipeline = DocumentPipeline(use_llm=False)
    
    # Process a document (replace with actual file path)
    # result = pipeline.process("path/to/invoice.pdf")
    # print(f"Document Type: {result['document_type']}")
    # print(f"Confidence: {result['metadata']['classification_confidence']:.2f}")


def example_with_llm():
    """Example with LLM enabled"""
    print("\n" + "=" * 60)
    print("Example 2: With LLM Support")
    print("=" * 60)
    
    # Initialize with LLM (requires API key)
    pipeline = DocumentPipeline(use_llm=True, llm_provider="openai")
    
    # LLM will be used as fallback when:
    # - Classification confidence < 0.5
    # - Extraction fails with regex
    # result = pipeline.process("path/to/document.pdf")


def example_known_type():
    """Example when document type is already known"""
    print("\n" + "=" * 60)
    print("Example 3: Known Document Type")
    print("=" * 60)
    
    pipeline = DocumentPipeline()
    
    # Skip classification if you already know the type
    # result = pipeline.process("invoice.pdf", document_type="invoice")
    # This is faster and more reliable


def example_validation():
    """Example showing validation results"""
    print("\n" + "=" * 60)
    print("Example 4: Validation")
    print("=" * 60)
    
    pipeline = DocumentPipeline()
    # result = pipeline.process("document.pdf")
    
    # Check validation
    # if result['validation']['is_valid']:
    #     print("✅ Document is valid")
    # else:
    #     print("❌ Validation errors:")
    #     for error in result['validation']['errors']:
    #         print(f"  - {error}")
    
    # if result['validation']['warnings']:
    #     print("⚠️  Warnings:")
    #     for warning in result['validation']['warnings']:
    #         print(f"  - {warning}")


def example_hash():
    """Example showing hash generation for CertiFi"""
    print("\n" + "=" * 60)
    print("Example 5: Hash Generation (for CertiFi)")
    print("=" * 60)
    
    pipeline = DocumentPipeline()
    # result = pipeline.process("document.pdf")
    
    # Get hashes for on-chain storage
    # text_hash = result['metadata']['text_hash']
    # canonical_hash = result['metadata']['canonical_hash']
    
    # print(f"Text Hash: {text_hash}")
    # print(f"Canonical Hash: {canonical_hash}")
    # 
    # The canonical hash is deterministic and can be used for:
    # - On-chain verification
    # - Deduplication
    # - Integrity checks


def example_json_output():
    """Example showing JSON output"""
    print("\n" + "=" * 60)
    print("Example 6: JSON Output")
    print("=" * 60)
    
    pipeline = DocumentPipeline()
    # result = pipeline.process("document.pdf")
    
    # Convert to JSON
    # json_output = pipeline.to_json(result)
    # print(json_output)
    
    # Save to file
    # output_file = "extracted_data.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(json_output)
    # print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    print("\n📚 CertiFi AI Pipeline - Usage Examples\n")
    
    example_basic()
    example_with_llm()
    example_known_type()
    example_validation()
    example_hash()
    example_json_output()
    
    print("\n" + "=" * 60)
    print("💡 Tip: Uncomment the code in each example to run it")
    print("=" * 60)
