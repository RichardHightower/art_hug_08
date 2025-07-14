"""Model optimization techniques for efficient deployment."""

import gc
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DEFAULT_SENTIMENT_MODEL, DEVICE


def benchmark_inference(model, tokenizer, texts, description=""):
    """Benchmark model inference speed and memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    start_time = time.time()

    # Run inference
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        memory_used = (peak_mem - start_mem) / 1024 / 1024  # MB
    else:
        memory_used = 0

    inference_time = end_time - start_time

    print(f"{description}")
    print(
        f"  Time: {inference_time:.3f}s ({inference_time/len(texts)*1000:.1f}ms per sample)"
    )
    print(f"  Memory: {memory_used:.1f}MB")
    print(f"  Throughput: {len(texts)/inference_time:.1f} samples/sec")

    return inference_time, memory_used


def demonstrate_optimization():
    """Demonstrate various optimization techniques."""

    print("Loading test data...")
    test_texts = [
        "This product is absolutely amazing!",
        "Terrible experience, would not recommend.",
        "Average quality, nothing special.",
        "Exceeded all my expectations!",
        "Complete waste of money.",
        "Pretty good for the price.",
        "Outstanding service and quality!",
        "Disappointing purchase.",
    ] * 4  # 32 samples

    print(f"Test set: {len(test_texts)} samples\n")

    print("1. Baseline Model Performance")
    print("-" * 40)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_SENTIMENT_MODEL)

    if DEVICE == "cuda":
        model = model.cuda()
    elif DEVICE == "mps":
        model = model.to("mps")

    # Baseline performance
    base_time, base_mem = benchmark_inference(
        model, tokenizer, test_texts, "FP32 Model"
    )

    # Model size
    param_size = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    )
    print(f"  Model size: {param_size:.1f}MB")

    print("\n2. Dynamic Quantization (INT8)")
    print("-" * 40)

    # Quantize model
    if DEVICE == "cpu":
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
            )

            q_time, q_mem = benchmark_inference(
                quantized_model, tokenizer, test_texts, "INT8 Quantized"
            )

            print(f"  Speedup: {base_time/q_time:.2f}x")
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e):
                print("  Skipping: Quantization not supported on this platform (macOS ARM)")
                print("  Note: Quantization requires x86_64 architecture or specific ARM builds")
            else:
                raise
    else:
        print("  Skipping (CPU only)")

    print("\n3. Batching Optimization")
    print("-" * 40)

    # Single sample inference
    single_times = []
    for text in test_texts[:8]:
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_times.append(time.time() - start)

    single_avg = sum(single_times) / len(single_times)
    print(f"Single inference: {single_avg*1000:.1f}ms per sample")

    # Batch inference
    batch_sizes = [1, 4, 8, 16, 32]
    for bs in batch_sizes:
        if bs > len(test_texts):
            continue

        batch_texts = test_texts[:bs]
        start = time.time()

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            _ = model(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time = time.time() - start
        per_sample = batch_time / bs * 1000

        print(
            f"Batch size {bs}: {per_sample:.1f}ms per sample (speedup: {single_avg*1000/per_sample:.1f}x)"
        )

    print("\n4. Half Precision (FP16)")
    print("-" * 40)

    if DEVICE == "cuda":
        model_fp16 = model.half()

        fp16_time, fp16_mem = benchmark_inference(
            model_fp16, tokenizer, test_texts, "FP16 Model"
        )

        print(f"  Speedup: {base_time/fp16_time:.2f}x")
        print(f"  Memory reduction: {(base_mem-fp16_mem)/base_mem*100:.1f}%")
    else:
        print("  Skipping (GPU only)")

    print("\n5. Model Pruning Simulation")
    print("-" * 40)

    # Simulate pruning by using a smaller model
    small_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading smaller model: {small_model_name}")

    small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    small_model = AutoModelForSequenceClassification.from_pretrained(
        small_model_name
    )

    if DEVICE == "cuda":
        small_model = small_model.cuda()
    elif DEVICE == "mps":
        small_model = small_model.to("mps")

    small_time, small_mem = benchmark_inference(
        small_model, small_tokenizer, test_texts, "DistilBERT (smaller)"
    )

    small_param_size = (
        sum(p.numel() * p.element_size() for p in small_model.parameters())
        / 1024
        / 1024
    )

    print(
        f"  Model size: {small_param_size:.1f}MB (reduction: {(param_size-small_param_size)/param_size*100:.1f}%)"
    )
    print(f"  Speedup: {base_time/small_time:.2f}x")

    print("\n6. Optimization Summary")
    print("-" * 40)
    print("| Technique | Speedup | Size Reduction | Use Case |")
    print("|-----------|---------|----------------|----------|")
    print("| Batching (32) | ~8x | 0% | GPU servers |")
    if DEVICE == "cpu" and 'q_time' in locals():
        print(f"| INT8 Quant | {base_time/q_time:.1f}x | 75% | CPU/Edge |")
    if DEVICE == "cuda" and 'fp16_time' in locals():
        print(f"| FP16 | {base_time/fp16_time:.1f}x | 50% | Modern GPUs |")
    print(
        f"| DistilBERT | {base_time/small_time:.1f}x | {(param_size-small_param_size)/param_size*100:.0f}% | General use |"
    )

    # Cleanup
    del model
    if "quantized_model" in locals():
        del quantized_model
    if "model_fp16" in locals():
        del model_fp16
    del small_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    demonstrate_optimization()
