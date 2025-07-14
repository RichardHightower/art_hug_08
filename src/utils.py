"""Utility functions for workflow examples."""

import time
from functools import wraps

import GPUtil
import psutil
import torch


def timer(func):
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.3f}s")
        return result

    return wrapper


# Alias for compatibility
timer_decorator = timer


def format_size(size_bytes):
    """Format byte size into human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_system_info():
    """Get current system resource usage."""
    info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
    }

    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        info.update(
            {
                "gpu_name": gpu.name,
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_temperature": gpu.temperature,
            }
        )

    return info


def format_number(num):
    """Format large numbers with K/M/B suffixes."""
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    else:
        return f"{num/1_000_000_000:.1f}B"


def calculate_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = (param_size + buffer_size) / 1024 / 1024  # MB

    return {
        "total_size_mb": total_size,
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024,
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def ensure_reproducibility(seed=42):
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MemoryTracker:
    """Context manager to track memory usage."""

    def __init__(self, device="cuda"):
        self.device = device
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated()

    def get_memory_used(self):
        """Get memory used in MB."""
        return (self.peak_memory - self.start_memory) / 1024 / 1024


def create_model_card(model_info):
    """Create a model card with key information."""
    card = f"""
# Model Card

## Model Details
- **Name**: {model_info.get('name', 'Unknown')}
- **Type**: {model_info.get('type', 'Unknown')}
- **Size**: {model_info.get('size_mb', 0):.1f}MB
- **Parameters**: {format_number(model_info.get('parameters', 0))}

## Performance
- **Inference Speed**: {model_info.get('inference_ms', 0):.1f}ms
- **Throughput**: {model_info.get('throughput', 0):.1f} samples/sec
- **Memory Usage**: {model_info.get('memory_mb', 0):.1f}MB

## Training
- **Dataset**: {model_info.get('dataset', 'Unknown')}
- **Epochs**: {model_info.get('epochs', 'Unknown')}
- **Batch Size**: {model_info.get('batch_size', 'Unknown')}

## Usage
```python
from transformers import pipeline

pipe = pipeline(
    '{model_info.get('task', 'text-classification')}',
    model='{model_info.get('name', 'model-name')}'
)
result = pipe("Your text here")
```
"""
    return card.strip()


def check_toxicity(texts):
    """
    Check toxicity of texts using evaluate library.
    
    Args:
        texts: List of texts to check
        
    Returns:
        List of toxicity scores (0-1, higher is more toxic)
    """
    try:
        from evaluate import load
        
        toxicity = load("toxicity")
        results = toxicity.compute(predictions=texts)
        return results["toxicity"]
    except Exception as e:
        print(f"Warning: Could not run toxicity check: {e}")
        # Return low toxicity scores as fallback
        return [0.0] * len(texts)


def filter_toxic_content(texts, threshold=0.5):
    """
    Filter out toxic content from a list of texts.
    
    Args:
        texts: List of texts to filter
        threshold: Toxicity threshold (0-1)
        
    Returns:
        List of non-toxic texts
    """
    toxicity_scores = check_toxicity(texts)
    filtered = []
    
    for text, score in zip(texts, toxicity_scores):
        if score < threshold:
            filtered.append(text)
        else:
            print(f"Filtered toxic content (score={score:.3f}): {text[:50]}...")
            
    return filtered


if __name__ == "__main__":
    # Test utilities
    print("System Info:")
    print(get_system_info())

    print("\nMemory Tracking Test:")
    with MemoryTracker() as tracker:
        # Simulate some GPU operations
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)

    print(f"Memory used: {tracker.get_memory_used():.2f}MB")
    
    # Test toxicity check
    print("\nToxicity Check Test:")
    test_texts = [
        "This is a nice product",
        "I hate this terrible thing",
        "The weather is beautiful today"
    ]
    
    toxicity_scores = check_toxicity(test_texts)
    for text, score in zip(test_texts, toxicity_scores):
        print(f"  '{text}' -> toxicity: {score:.3f}")
    
    # Test filtering
    print("\nFiltering Test:")
    filtered = filter_toxic_content(test_texts, threshold=0.7)
    print(f"Kept {len(filtered)}/{len(test_texts)} texts")
