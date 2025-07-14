"""
Flash Attention demonstration for efficient transformer inference.

This module shows how to use Flash Attention and other attention optimizations
to significantly speed up transformer model inference on GPUs.
"""

import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.config import Config
from src.utils import format_size, timer_decorator


class FlashAttentionDemo:
    """Demonstrates Flash Attention and attention optimization techniques."""

    def __init__(self):
        """Initialize Flash Attention demo."""
        self.device = Config.DEVICE
        self.has_flash_attn = self._check_flash_attention()

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            # Check for flash_attn package
            import flash_attn

            print("✅ Flash Attention package found")

            # Check CUDA capability
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 8:  # Ampere or newer
                    print(
                        f"✅ GPU supports Flash Attention (compute capability {capability[0]}.{capability[1]})"
                    )
                    return True
                else:
                    print(
                        f"⚠️  GPU compute capability {capability[0]}.{capability[1]} < 8.0 (Ampere)"
                    )
            return False
        except ImportError:
            print(
                "⚠️  Flash Attention not installed. Install with: pip install flash-attn"
            )
            return False

    @timer_decorator
    def compare_attention_methods(
        self, model_name: str = "microsoft/phi-2", sequence_length: int = 512, batch_size: int = 8
    ) -> dict[str, Any]:
        """
        Compare different attention implementations.

        Args:
            model_name: Model to test
            sequence_length: Input sequence length
            batch_size: Batch size for testing

        Returns:
            Comparison metrics
        """
        print(f"\n🔬 Comparing Attention Methods for {model_name}")
        print(f"   Sequence length: {sequence_length}, Batch size: {batch_size}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create dummy input
        dummy_text = " ".join(["Sample text"] * (sequence_length // 10))
        inputs = tokenizer(
            [dummy_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sequence_length,
        ).to(self.device)

        results = {}

        # 1. Standard Attention
        print("\n📊 Testing Standard Attention...")
        model_standard = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",
        )

        # Warm up
        with torch.no_grad():
            _ = model_standard(**inputs)

        # Benchmark
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_standard(**inputs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        standard_time = time.time() - start_time
        standard_mem = (
            torch.cuda.memory_allocated() - start_mem
            if self.device.type == "cuda"
            else 0
        )

        results["standard"] = {
            "time": f"{standard_time:.3f}s",
            "memory": format_size(standard_mem),
            "throughput": f"{(10 * batch_size) / standard_time:.1f} samples/s",
        }

        del model_standard
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # 2. Flash Attention (if available)
        if self.has_flash_attn and self.device.type == "cuda":
            print("\n⚡ Testing Flash Attention...")

            # Load model with Flash Attention
            model_flash = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                use_flash_attention_2=True,  # Enable Flash Attention 2
            )

            # Warm up
            with torch.no_grad():
                _ = model_flash(**inputs)

            # Benchmark
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model_flash(**inputs)

            torch.cuda.synchronize()
            flash_time = time.time() - start_time
            flash_mem = torch.cuda.memory_allocated() - start_mem

            results["flash_attention"] = {
                "time": f"{flash_time:.3f}s",
                "memory": format_size(flash_mem),
                "throughput": f"{(10 * batch_size) / flash_time:.1f} samples/s",
                "speedup": f"{standard_time / flash_time:.2f}x",
            }

            del model_flash
            torch.cuda.empty_cache()

        # 3. Scaled Dot Product Attention (SDPA) - PyTorch 2.0+
        if torch.__version__ >= "2.0.0":
            print("\n🔧 Testing Scaled Dot Product Attention (SDPA)...")

            model_sdpa = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                device_map="auto",
                attn_implementation="sdpa",  # Use PyTorch's SDPA
            )

            # Warm up
            with torch.no_grad():
                _ = model_sdpa(**inputs)

            # Benchmark
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model_sdpa(**inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            sdpa_time = time.time() - start_time
            sdpa_mem = (
                torch.cuda.memory_allocated() - start_mem
                if self.device.type == "cuda"
                else 0
            )

            results["sdpa"] = {
                "time": f"{sdpa_time:.3f}s",
                "memory": format_size(sdpa_mem),
                "throughput": f"{(10 * batch_size) / sdpa_time:.1f} samples/s",
                "speedup": f"{standard_time / sdpa_time:.2f}x",
            }

            del model_sdpa
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Print results
        print("\n📊 Attention Method Comparison:")
        print(
            f"{'Method':<20} {'Time':<10} {'Memory':<15} {'Throughput':<20} {'Speedup':<10}"
        )
        print("-" * 85)
        for method, metrics in results.items():
            speedup = metrics.get("speedup", "1.00x")
            print(
                f"{method:<20} {metrics['time']:<10} {metrics['memory']:<15} "
                f"{metrics['throughput']:<20} {speedup:<10}"
            )

        return results

    def demonstrate_memory_efficient_attention(
        self, model_name: str = "gpt2-medium", max_length: int = 1024
    ) -> dict[str, Any]:
        """
        Demonstrate memory-efficient attention for long sequences.

        Args:
            model_name: Model to use
            max_length: Maximum sequence length

        Returns:
            Memory usage comparison
        """
        print("\n💾 Memory-Efficient Attention Demo")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Create long input
        long_text = " ".join(["This is a long text sequence."] * (max_length // 10))
        inputs = tokenizer(
            long_text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        results = {}

        # Test different configurations
        configs = [
            {"name": "Standard FP32", "dtype": torch.float32, "use_cache": True},
            {"name": "FP16", "dtype": torch.float16, "use_cache": True},
            {"name": "FP16 No Cache", "dtype": torch.float16, "use_cache": False},
        ]

        if self.has_flash_attn and self.device.type == "cuda":
            configs.append(
                {
                    "name": "Flash Attention",
                    "dtype": torch.float16,
                    "use_cache": False,
                    "use_flash": True,
                }
            )

        for config in configs:
            print(f"\n📊 Testing {config['name']}...")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

            # Load model with config
            model_kwargs = {
                "torch_dtype": config["dtype"],
                "device_map": "auto",
                "use_cache": config["use_cache"],
            }

            if config.get("use_flash"):
                model_kwargs["use_flash_attention_2"] = True

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
                peak_mem = torch.cuda.memory_allocated() - start_mem
            else:
                peak_mem = 0

            results[config["name"]] = {
                "peak_memory": format_size(peak_mem),
                "dtype": str(config["dtype"]).split(".")[-1],
                "use_cache": config["use_cache"],
            }

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Print comparison
        print("\n📊 Memory Usage Comparison:")
        print(
            f"{'Configuration':<20} {'Peak Memory':<15} {'Data Type':<10} {'KV Cache':<10}"
        )
        print("-" * 65)
        for name, metrics in results.items():
            print(
                f"{name:<20} {metrics['peak_memory']:<15} {metrics['dtype']:<10} "
                f"{'Yes' if metrics['use_cache'] else 'No':<10}"
            )

        return results

    def benchmark_batch_processing(
        self,
        model_name: str = "distilbert-base-uncased",
        batch_sizes: list[int] = [1, 4, 8, 16, 32],
    ) -> dict[str, Any]:
        """
        Benchmark attention performance with different batch sizes.

        Args:
            model_name: Model to benchmark
            batch_sizes: List of batch sizes to test

        Returns:
            Benchmark results
        """
        print("\n📈 Batch Processing Benchmark with Attention Optimizations")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        results = []

        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")

            # Create batch
            texts = ["This is a sample text for benchmarking."] * batch_size
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            # Warm up
            with torch.no_grad():
                _ = model(**inputs)

            # Benchmark
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(**inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            total_time = time.time() - start_time

            results.append(
                {
                    "batch_size": batch_size,
                    "total_time": f"{total_time:.3f}s",
                    "throughput": f"{(20 * batch_size) / total_time:.1f} samples/s",
                    "avg_latency": f"{(total_time / 20) * 1000:.1f}ms",
                }
            )

        # Print results
        print("\n📊 Batch Processing Results:")
        print(
            f"{'Batch Size':<12} {'Total Time':<12} {'Throughput':<20} {'Avg Latency':<15}"
        )
        print("-" * 70)
        for result in results:
            print(
                f"{result['batch_size']:<12} {result['total_time']:<12} "
                f"{result['throughput']:<20} {result['avg_latency']:<15}"
            )

        return {"batch_results": results}


def demonstrate_flash_attention():
    """Demonstrate Flash Attention and attention optimizations."""
    print("=" * 80)
    print("⚡ FLASH ATTENTION DEMONSTRATION")
    print("=" * 80)

    demo = FlashAttentionDemo()

    # 1. Compare attention methods
    print("\n1️⃣ Attention Method Comparison")
    attention_results = demo.compare_attention_methods(
        sequence_length=512, batch_size=4
    )

    # 2. Memory-efficient attention
    print("\n2️⃣ Memory-Efficient Attention")
    memory_results = demo.demonstrate_memory_efficient_attention()

    # 3. Batch processing benchmark
    print("\n3️⃣ Batch Processing Performance")
    batch_results = demo.benchmark_batch_processing()

    print("\n" + "=" * 80)
    print("📊 ATTENTION OPTIMIZATION SUMMARY")
    print("=" * 80)

    if demo.has_flash_attn:
        print("\n✅ Flash Attention is available and demonstrated!")
        print("   - Significantly reduced memory usage")
        print("   - Faster processing for long sequences")
        print("   - Better scaling with batch size")
    else:
        print("\n⚠️  Flash Attention not available, but demonstrated:")
        print("   - PyTorch SDPA optimization")
        print("   - FP16 memory savings")
        print("   - Batch processing optimization")

    print("\n💡 Key Takeaways:")
    print("   - Attention optimization can provide 2-4x speedup")
    print("   - Memory usage can be reduced by 50-75%")
    print("   - Larger batches are more efficient per sample")
    print("   - Flash Attention excels with long sequences")


if __name__ == "__main__":
    demonstrate_flash_attention()
