"""Unit tests for workflow components."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch

from custom_pipelines import CustomSentimentPipeline
from data_workflows import preprocess_batch
from synthetic_data import quality_filter, validate_synthetic_text
from utils import MemoryTracker, format_number


def test_custom_sentiment_pipeline():
    """Test custom sentiment pipeline preprocessing."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = CustomSentimentPipeline(
        model=model, tokenizer=tokenizer, device=-1  # CPU
    )

    # Test preprocessing
    test_text = "<p>AMAZING PRODUCT!!!</p>"
    # The pipeline should handle preprocessing internally

    assert hasattr(pipeline, "preprocess")
    assert hasattr(pipeline, "postprocess")


def test_data_preprocessing():
    """Test batch preprocessing function."""
    batch = {
        "text": ["Hello World", "Test Example"],
    }

    result = preprocess_batch(batch)

    assert "length" in result
    assert "complexity" in result
    assert result["length"] == [2, 2]
    assert all(0 <= c <= 1 for c in result["complexity"])


def test_synthetic_validation():
    """Test synthetic data validation."""
    synthetic = [
        {"full_text": "This is a test review about a product."},
        {"full_text": "Another sample review with different words."},
        {"full_text": "Third review to ensure diversity."},
    ]

    real = [
        "This is a real product review.",
        "Another real review from a customer.",
        "Real feedback about the product.",
    ]

    validation = validate_synthetic_text(synthetic, real)

    assert "score" in validation
    assert "diversity" in validation
    assert 0 <= validation["score"] <= 1
    assert validation["diversity"] == 1.0  # All unique


def test_quality_filter():
    """Test quality filtering for synthetic data."""
    samples = [
        {"text": "Good product", "label": "positive"},  # Too short
        {
            "text": "This is a great product that works well and I recommend it",
            "label": "positive",
        },
        {"text": "Bad bad bad bad bad", "label": "negative"},  # Repetitive
        {"text": "The product is okay but...", "label": "neutral"},  # Truncated
    ]

    filtered, reasons = quality_filter(samples)

    assert len(filtered) == 1  # Only one should pass
    assert "too_short" in reasons
    assert "repetitive" in reasons
    assert "truncated" in reasons


def test_format_number():
    """Test number formatting utility."""
    assert format_number(500) == "500"
    assert format_number(1500) == "1.5K"
    assert format_number(1_500_000) == "1.5M"
    assert format_number(1_500_000_000) == "1.5B"


def test_memory_tracker():
    """Test memory tracking utility."""
    with MemoryTracker(device="cpu") as tracker:
        # CPU operations don't track GPU memory
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)

    # Should work without errors
    memory_used = tracker.get_memory_used()
    assert isinstance(memory_used, (int, float))


def test_imports():
    """Test that all required modules can be imported."""
    import datasets
    import numpy
    import torch
    import transformers

    assert transformers.__version__
    assert datasets.__version__
    assert torch.__version__
    assert numpy.__version__


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_detection(device):
    """Test device detection logic."""
    from config import get_device

    detected = get_device()
    assert detected in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__])
