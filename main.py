"""
Main entry point for CertiFi AI Pipeline
Example usage
"""

from pathlib import Path
from pipeline.orchestrator import DocumentPipeline


def main():
    """Example usage of the pipeline"""
    
    # Initialize pipeline
    pipeline = DocumentPipeline(use_llm=False)  # Set to True when you have API keys
    
    # Example: Process a document
    file_path = input("Enter path to document: ").strip()
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"\nProcessing: {file_path}")
    print("-" * 50)
    
    # Process document
    result = pipeline.process(file_path)
    
    # Print results - OPERATIONAL FOCUS
    print(f"\n✅ Extraction Success: {result['success']}")
    print(f"📄 Document Type: {result['document_type']}")
    print(f"🎯 Classification Confidence: {result['metadata'].get('classification_confidence', 0):.2f}")
    
    # THE REAL DECISION (from DecisionEngine)
    print(f"\n{'='*50}")
    print("🔐 CERTIFICATION DECISION")
    print(f"{'='*50}")
    print(f"  Certification Ready: {'✅ YES' if result.get('certification_ready') else '❌ NO'}")
    print(f"  Human Review Required: {'⚠️  YES' if result.get('human_review_required') else '✅ NO'}")
    
    # Risk level and profile
    if result.get('risk_level'):
        risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        print(f"  Risk Level: {risk_emoji.get(result['risk_level'], '⚪')} {result['risk_level'].upper()}")
    
    if result.get('certification_profile'):
        print(f"  Certification Profile: {result['certification_profile']}")
    
    # Decision details
    if result.get('metadata', {}).get('decision'):
        decision = result['metadata']['decision']
        print(f"\n  Decision Reason: {decision.get('reason', 'unknown')}")
        if decision.get('details'):
            print(f"  Details: {decision['details']}")
    
    if result.get('data'):
        print(f"\n  Overall Confidence: {result['data'].confidence:.2f} (adjusted, never 1.0)")
        print(f"  Classification Confidence: {result['metadata'].get('classification_confidence', 0):.2f}")
        
        if hasattr(result['data'], 'field_confidence') and result['data'].field_confidence:
            print(f"\n  Field Confidence:")
            for field, conf in result['data'].field_confidence.items():
                print(f"    - {field}: {conf:.2f}")
        
        if hasattr(result['data'], 'trusted_source'):
            print(f"  Trusted Source: {result['data'].trusted_source or 'N/A'}")
        
        if hasattr(result['data'], 'missing_fields') and result['data'].missing_fields:
            print(f"  Missing Fields: {', '.join(result['data'].missing_fields)}")
    
    if result.get('validation'):
        print(f"\n📋 Validation Details:")
        if result['validation'].get('errors'):
            print(f"  ❌ Errors:")
            for error in result['validation']['errors']:
                print(f"    - {error}")
        if result['validation'].get('warnings'):
            print(f"  ⚠️  Warnings:")
            for warning in result['validation']['warnings']:
                print(f"    - {warning}")
        if result['validation'].get('reason'):
            print(f"  Reason: {result['validation']['reason']}")
    
    if result['data']:
        print(f"\n📊 Extracted Data:")
        print(result['data'].model_dump_json(indent=2))
    
    # Hash generation status
    if result['metadata'].get('canonical_hash'):
        print(f"\n🔐 Canonical Hash (for on-chain): {result['metadata']['canonical_hash']}")
    elif result.get('certification_ready') == False:
        print(f"\n🔐 Canonical Hash: NOT GENERATED (document not certification ready)")
        if result['metadata'].get('hash_generation_blocked'):
            print(f"   Reason: {result['metadata']['hash_generation_blocked']}")
    
    if result['errors']:
        print(f"\n❌ Pipeline Errors: {result['errors']}")
    
    # Save to JSON
    output_file = Path(file_path).stem + "_extracted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pipeline.to_json(result))
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
