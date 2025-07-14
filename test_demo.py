#!/usr/bin/env python
"""Test basic demo functionality."""

print("Testing basic pipeline functionality...")

from transformers import pipeline

try:
    # Create a simple pipeline with modern model
    clf = pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest'
    )
    
    # Test it
    text = "I love this new Hugging Face library!"
    result = clf(text)
    
    print(f"\nText: {text}")
    print(f"Result: {result}")
    
    # Test batch processing
    texts = [
        "This is amazing!",
        "I'm not sure about this...",
        "Terrible experience."
    ]
    batch_results = clf(texts)
    
    print("\nBatch processing results:")
    for text, result in zip(texts, batch_results):
        print(f"  {text}: {result['label']} ({result['score']:.3f})")
    
    print("\n✅ Basic functionality works!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying fallback model...")
    
    try:
        # Fallback to default model
        clf = pipeline('sentiment-analysis')
        result = clf("Test with default model")
        print(f"✅ Fallback model works: {result}")
    except Exception as fallback_error:
        print(f"❌ Fallback also failed: {fallback_error}")
        exit(1)