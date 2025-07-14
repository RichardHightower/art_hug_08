#!/usr/bin/env python
"""Test environment setup."""

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import transformers
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
    
    try:
        import torch
        print("✓ torch imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
    
    try:
        import datasets
        print("✓ datasets imported successfully")
    except ImportError as e:
        print(f"✗ datasets import failed: {e}")
    
    try:
        import diffusers
        print("✓ diffusers imported successfully")
    except ImportError as e:
        print(f"✗ diffusers import failed: {e}")
    
    try:
        import peft
        print("✓ peft imported successfully")
    except ImportError as e:
        print(f"✗ peft import failed: {e}")
    
    try:
        import sentencepiece
        print("✓ sentencepiece imported successfully")
    except ImportError as e:
        print(f"✗ sentencepiece import failed: {e}")
    
    try:
        import accelerate
        print("✓ accelerate imported successfully")
    except ImportError as e:
        print(f"✗ accelerate import failed: {e}")
    
    try:
        import bitsandbytes
        print("✓ bitsandbytes imported successfully")
    except ImportError as e:
        print(f"✗ bitsandbytes import failed: {e}")
    
    try:
        import evaluate
        print("✓ evaluate imported successfully")
    except ImportError as e:
        print(f"✗ evaluate import failed: {e}")
    
    try:
        from transformers import BitsAndBytesConfig
        print("✓ BitsAndBytesConfig imported successfully")
    except ImportError as e:
        print(f"✗ BitsAndBytesConfig import failed: {e}")
    
    print("\nAll imports tested!")
    

def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("\nTesting basic pipeline...")
    
    from transformers import pipeline
    
    try:
        # Create a simple sentiment analysis pipeline with modern model
        clf = pipeline(
            'sentiment-analysis', 
            model='cardiffnlp/twitter-roberta-base-sentiment-latest'
        )
        
        # Test it
        result = clf("I love Hugging Face!")
        print(f"Pipeline test result: {result}")
        print("✓ Modern model works!")
    except Exception as e:
        print(f"Modern model failed: {e}")
        print("Trying default model...")
        
        # Fallback to default
        clf = pipeline('sentiment-analysis')
        result = clf("I love Hugging Face!")
        print(f"Default pipeline result: {result}")
        print("✓ Default model works!")
    

if __name__ == "__main__":
    test_imports()
    test_basic_pipeline()
    print("\n✅ Environment setup complete!")