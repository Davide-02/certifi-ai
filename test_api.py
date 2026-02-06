"""
Script di test per l'API CertiFi AI

Testa l'endpoint /analyze con una richiesta di esempio
"""

import requests
import json
import sys
from pathlib import Path

API_URL = "http://localhost:8000/analyze"

def test_health():
    """Test health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_analyze(file_path: str):
    """Test analyze endpoint"""
    if not Path(file_path).exists():
        print(f"❌ File non trovato: {file_path}")
        return False
    
    # Calcola hash del file
    import hashlib
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    print(f"\n📄 Test analisi documento: {file_path}")
    print(f"📊 Hash: {file_hash[:16]}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            data = {
                'document_id': 'test_doc_001',
                'hash': f'sha256:{file_hash}',
                'requested_tasks': 'classify,extract,claims,holder,compliance_score',
                'ai_version': 'v1.0'
            }
            
            print("\n📤 Invio richiesta...")
            response = requests.post(API_URL, files=files, data=data, timeout=60)
            
            print(f"\n📥 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Analisi completata!")
                print(json.dumps(result, indent=2, default=str))
                return True
            else:
                print(f"\n❌ Errore: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Errore: Impossibile connettersi all'API")
        print("   Assicurati che l'API sia in esecuzione: python api.py")
        return False
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Test CertiFi AI API")
    print("=" * 50)
    
    # Test health check
    if not test_health():
        print("\n⚠️  L'API non risponde. Avviala con: python api.py")
        sys.exit(1)
    
    # Test analyze (se viene fornito un file)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        test_analyze(file_path)
    else:
        print("\n💡 Per testare l'analisi di un documento:")
        print("   python test_api.py /path/to/document.pdf")
