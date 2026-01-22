"""
Simple test script for the pipeline
Run this to verify installation and basic functionality
"""

from pipeline.classifier import DocumentClassifier
from pipeline.preprocessing import DocumentPreprocessor


def test_classifier():
    """Test document classification"""
    print("Testing Document Classifier...")
    
    classifier = DocumentClassifier()
    
    # Test invoice
    invoice_text = """
    FATTURA N. 123/2024
    Data: 15/01/2024
    Cliente: Mario Rossi
    Totale: € 1.000,00
    IVA: € 220,00
    """
    
    result = classifier.classify(invoice_text)
    assert result['type'] == 'invoice', f"Expected 'invoice', got '{result['type']}'"
    print(f"✅ Invoice classification: {result['type']} (confidence: {result['confidence']:.2f})")
    
    # Test diploma
    diploma_text = """
    UNIVERSITÀ DEGLI STUDI DI MILANO
    DIPLOMA DI LAUREA
    Studente: Luigi Bianchi
    CFU: 180
    Data di laurea: 20/07/2023
    """
    
    result = classifier.classify(diploma_text)
    assert result['type'] == 'diploma', f"Expected 'diploma', got '{result['type']}'"
    print(f"✅ Diploma classification: {result['type']} (confidence: {result['confidence']:.2f})")
    
    # Test ID
    id_text = """
    CARTA DI IDENTITÀ
    Nome: Giuseppe
    Cognome: Verdi
    Data di nascita: 01/01/1990
    Codice fiscale: VRDGPP90A01H501X
    """
    
    result = classifier.classify(id_text)
    assert result['type'] == 'id', f"Expected 'id', got '{result['type']}'"
    print(f"✅ ID classification: {result['type']} (confidence: {result['confidence']:.2f})")


def test_preprocessing():
    """Test text preprocessing"""
    print("\nTesting Text Preprocessing...")
    
    preprocessor = DocumentPreprocessor()
    
    dirty_text = """
    Questo    è   un    testo    con    molti    spazi
    
    
    E anche molte righe vuote.
    """
    
    cleaned = preprocessor.normalize_text(dirty_text)
    assert "   " not in cleaned, "Multiple spaces not removed"
    assert "\n\n\n" not in cleaned, "Multiple newlines not removed"
    print("✅ Text normalization working")


def test_imports():
    """Test that all modules can be imported"""
    print("\nTesting Imports...")
    
    try:
        from pipeline.ocr import TextExtractor
        from pipeline.extractor import InformationExtractor
        from pipeline.validators import DocumentValidator
        from pipeline.orchestrator import DocumentPipeline
        from pipeline.schemas import InvoiceSchema, DiplomaSchema, IDDocumentSchema
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("CertiFi AI Pipeline - Basic Tests")
    print("=" * 60)
    
    try:
        test_imports()
        test_classifier()
        test_preprocessing()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("  1. Install Tesseract OCR: brew install tesseract tesseract-lang")
        print("  2. Test with a real document: python main.py")
        print("  3. Configure LLM (optional): Set OPENAI_API_KEY in .env")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
