"""Efficient data handling with Hugging Face Datasets."""

import time

import numpy as np
import psutil
from datasets import Dataset, load_dataset

from src.config import CACHE_DIR, NUM_WORKERS


def measure_memory():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def preprocess_batch(batch):
    """Preprocess a batch of examples."""
    # Lowercase text
    batch["text"] = [text.lower() for text in batch["text"]]

    # Add text length
    batch["length"] = [len(text.split()) for text in batch["text"]]

    # Add complexity score (simple example)
    batch["complexity"] = [
        len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0
        for text in batch["text"]
    ]

    return batch


def demonstrate_data_workflows():
    """Demonstrate efficient data handling techniques."""

    print("1. Standard Data Loading")
    print("-" * 40)

    # Load small dataset
    start_mem = measure_memory()
    start_time = time.time()

    dataset = load_dataset("imdb", split="train[:1000]", cache_dir=CACHE_DIR)

    load_time = time.time() - start_time
    load_mem = measure_memory() - start_mem

    print(f"Loaded {len(dataset)} examples")
    print(f"Time: {load_time:.2f}s, Memory: {load_mem:.2f}MB")
    print(f"First example: {dataset[0]['text'][:100]}...")

    print("\n2. Efficient Batch Processing")
    print("-" * 40)

    # Process without batching (slow)
    start_time = time.time()
    dataset_slow = dataset.map(
        lambda x: {"text": x["text"].lower(), "length": len(x["text"].split())},
        desc="Processing (no batching)",
    )
    no_batch_time = time.time() - start_time

    # Process with batching (fast)
    start_time = time.time()
    dataset_fast = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=100,
        num_proc=NUM_WORKERS,
        desc="Processing (batched)",
    )
    batch_time = time.time() - start_time

    print(f"Without batching: {no_batch_time:.2f}s")
    print(f"With batching: {batch_time:.2f}s")
    print(f"Speedup: {no_batch_time/batch_time:.2f}x")

    print("\n3. Filtering and Selection")
    print("-" * 40)

    # Filter long reviews
    long_reviews = dataset_fast.filter(
        lambda x: x["length"] > 100, desc="Filtering long reviews"
    )

    print(f"Original dataset: {len(dataset_fast)} examples")
    print(f"After filtering: {len(long_reviews)} examples")

    # Select specific columns
    slim_dataset = long_reviews.select_columns(["text", "length"])
    print(f"Columns after selection: {slim_dataset.column_names}")

    print("\n4. Streaming Large Datasets")
    print("-" * 40)

    # Stream dataset without loading into memory
    print("Loading Wikipedia (streaming mode)...")
    start_mem = measure_memory()

    streaming_dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    # Process first 100 examples
    processed_count = 0
    total_length = 0

    for example in streaming_dataset:
        processed_count += 1
        total_length += len(example["text"].split())

        if processed_count >= 100:
            break

    stream_mem = measure_memory() - start_mem

    print(f"Processed {processed_count} Wikipedia articles")
    print(f"Average article length: {total_length/processed_count:.0f} words")
    print(f"Memory used: {stream_mem:.2f}MB (vs ~6GB for full dataset)")

    print("\n5. Creating Custom Datasets")
    print("-" * 40)

    # Create dataset from Python objects
    custom_data = {
        "text": [
            "This is a positive review.",
            "This is a negative review.",
            "This is a neutral comment.",
        ],
        "label": [1, 0, 2],
        "source": ["web", "app", "email"],
    }

    custom_dataset = Dataset.from_dict(custom_data)
    print(f"Custom dataset: {custom_dataset}")

    # Add computed columns
    def add_features(example):
        example["num_words"] = len(example["text"].split())
        example["has_punctuation"] = any(c in example["text"] for c in ".,!?")
        return example

    custom_dataset = custom_dataset.map(add_features)
    print(f"With features: {custom_dataset}")
    print(f"First example: {custom_dataset[0]}")

    print("\n6. Dataset Statistics")
    print("-" * 40)

    # Compute statistics efficiently
    lengths = dataset_fast["length"]
    complexities = dataset_fast["complexity"]

    print("Text length statistics:")
    print(f"  Mean: {np.mean(lengths):.1f} words")
    print(f"  Std: {np.std(lengths):.1f} words")
    print(f"  Min: {np.min(lengths)} words")
    print(f"  Max: {np.max(lengths)} words")

    print("\nComplexity statistics:")
    print(f"  Mean: {np.mean(complexities):.3f}")
    print(f"  Std: {np.std(complexities):.3f}")


if __name__ == "__main__":
    demonstrate_data_workflows()
