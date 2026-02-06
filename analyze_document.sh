#!/bin/bash

# Script per analizzare un documento tramite API CertiFi AI
# Uso: ./analyze_document.sh /path/to/document.pdf

set -e

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Verifica argomenti
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Errore: Specifica il percorso del documento${NC}"
    echo ""
    echo "Uso: $0 <file_documento> [document_id] [tasks]"
    echo ""
    echo "Esempi:"
    echo "  $0 document.pdf"
    echo "  $0 document.pdf doc_123"
    echo "  $0 document.pdf doc_123 classify,extract,claims,holder,compliance_score"
    exit 1
fi

FILE_PATH="$1"
DOCUMENT_ID="${2:-doc_$(date +%s)}"
REQUESTED_TASKS="${3:-classify,extract,claims,holder,compliance_score}"

# Verifica che il file esista
if [ ! -f "$FILE_PATH" ]; then
    echo -e "${RED}❌ Errore: File non trovato: $FILE_PATH${NC}"
    exit 1
fi

# Calcola hash SHA256 del file
echo -e "${YELLOW}📄 File: $FILE_PATH${NC}"
echo -e "${YELLOW}🆔 Document ID: $DOCUMENT_ID${NC}"
echo -e "${YELLOW}📋 Tasks: $REQUESTED_TASKS${NC}"
echo ""
echo -e "${YELLOW}🔐 Calcolo hash SHA256...${NC}"

if command -v shasum &> /dev/null; then
    # macOS
    FILE_HASH=$(shasum -a 256 "$FILE_PATH" | awk '{print $1}')
elif command -v sha256sum &> /dev/null; then
    # Linux
    FILE_HASH=$(sha256sum "$FILE_PATH" | awk '{print $1}')
else
    echo -e "${RED}❌ Errore: shasum o sha256sum non trovato${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Hash: ${FILE_HASH:0:16}...${NC}"
echo ""

# URL API
API_URL="http://localhost:8000/analyze"

# Verifica che l'API sia in esecuzione
echo -e "${YELLOW}🔍 Verifica connessione API...${NC}"
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Errore: API non raggiungibile su http://localhost:8000${NC}"
    echo -e "${YELLOW}💡 Avvia l'API con: python api.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API raggiungibile${NC}"
echo ""

# Invia richiesta
echo -e "${YELLOW}📤 Invio richiesta all'API...${NC}"
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
  -F "document_id=$DOCUMENT_ID" \
  -F "hash=sha256:$FILE_HASH" \
  -F "requested_tasks=$REQUESTED_TASKS" \
  -F "ai_version=v1.0" \
  -F "file=@$FILE_PATH")

# Estrai status code e body
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$HTTP_CODE" -eq 200 ]; then
    echo -e "${GREEN}✅ Analisi completata con successo!${NC}"
    echo ""
    
    # Formatta output JSON (se jq è disponibile)
    if command -v jq &> /dev/null; then
        echo "$BODY" | jq '.'
    else
        echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Estrai informazioni chiave
    echo ""
    echo -e "${GREEN}📊 Riepilogo:${NC}"
    echo "$BODY" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Document Family: {data.get('document_family', 'N/A')}\")
    print(f\"  Document Type: {data.get('document_type', 'N/A')}\")
    if 'compliance_score' in data and data['compliance_score']:
        print(f\"  Compliance Score: {data['compliance_score']:.2f}\")
    if 'anomalies' in data and data['anomalies']:
        print(f\"  ⚠️  Anomalie: {len(data['anomalies'])}\")
        for a in data['anomalies']:
            print(f\"     - {a}\")
except:
    pass
" 2>/dev/null || echo "  (Usa jq o python per formattare meglio)"
    
else
    echo -e "${RED}❌ Errore HTTP $HTTP_CODE${NC}"
    echo ""
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi
