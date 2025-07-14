"""
Advanced quantization techniques including INT4 for large language models.

This module demonstrates INT4, INT8, and other quantization methods for
deploying large models with minimal memory footprint and maximum performance.
"""

import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import Config
from src.utils import format_size, timer_decorator


class AdvancedQuantization:
    """Advanced quantization techniques for large models."""

    def __init__(self):
        """Initialize quantization demo."""
        self.device = Config.DEVICE
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import bitsandbytes as bnb

            print("✅ bitsandbytes library found")
        except ImportError:
            print(
                "⚠️  bitsandbytes not installed. Install with: pip install bitsandbytes"
            )

    @timer_decorator
    def quantize_model_int4(
        self,
        model_name: str = "gpt2",
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",
    ) -> dict[str, Any]:
        """
        Quantize model to INT4 using bitsandbytes.

        Args:
            model_name: Model to quantize
            compute_dtype: Compute dtype for forward pass
            quant_type: Quantization type (nf4 or fp4)

        Returns:
            Quantization metrics
        """
        print(f"\n🔢 Quantizing {model_name} to INT4 ({quant_type})...")

        # Configure INT4 quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,  # nf4 or fp4
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Nested quantization
        )

        # Track memory before loading
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

        # Load model with INT4 quantization
        start_time = time.time()
        model_int4 = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        load_time = time.time() - start_time

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Track memory after loading
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            memory_used = mem_after - mem_before
        else:
            memory_used = 0

        # Test inference
        test_text = "The future of AI is"
        inputs = tokenizer(test_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            start_time = time.time()
            outputs = model_int4.generate(**inputs, max_new_tokens=20)
            inference_time = time.time() - start_time

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate model size (approximate)
        param_count = sum(p.numel() for p in model_int4.parameters())
        int4_size = param_count * 0.5  # 4 bits = 0.5 bytes per parameter
        fp32_size = param_count * 4  # 32 bits = 4 bytes per parameter

        metrics = {
            "quantization": f"INT4 ({quant_type})",
            "load_time": f"{load_time:.2f}s",
            "memory_used": format_size(memory_used),
            "inference_time": f"{inference_time:.3f}s",
            "approx_model_size": format_size(int4_size),
            "size_reduction": f"{(1 - int4_size/fp32_size)*100:.1f}%",
            "generated_text": generated,
        }

        print("\n✅ INT4 Quantization Complete:")
        for key, value in metrics.items():
            if key != "generated_text":
                print(f"  {key}: {value}")

        return metrics, model_int4

    def compare_quantization_methods(
        self, model_name: str = "gpt2", test_sequences: int = 10
    ) -> dict[str, Any]:
        """
        Compare different quantization methods.

        Args:
            model_name: Model to test
            test_sequences: Number of sequences for benchmarking

        Returns:
            Comparison results
        """
        print(f"\n🔬 Comparing Quantization Methods for {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test inputs
        test_texts = [
            f"Sample text {i} for quantization testing." for i in range(test_sequences)
        ]

        results = {}

        # 1. FP32 Baseline
        print("\n📊 Testing FP32 (Baseline)...")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        model_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto"
        )

        results["FP32"] = self._benchmark_model(model_fp32, tokenizer, test_texts)
        del model_fp32

        # 2. FP16
        print("\n📊 Testing FP16...")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

            model_fp16 = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

            results["FP16"] = self._benchmark_model(model_fp16, tokenizer, test_texts)
            del model_fp16

        # 3. INT8
        print("\n📊 Testing INT8...")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        bnb_config_int8 = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
        )

        model_int8 = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config_int8, device_map="auto"
        )

        results["INT8"] = self._benchmark_model(model_int8, tokenizer, test_texts)
        del model_int8

        # 4. INT4 (NF4)
        print("\n📊 Testing INT4 (NF4)...")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        bnb_config_nf4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            bnb_4bit_use_double_quant=True,
        )

        model_nf4 = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config_nf4, device_map="auto"
        )

        results["INT4-NF4"] = self._benchmark_model(model_nf4, tokenizer, test_texts)
        del model_nf4

        # Print comparison
        print("\n📊 Quantization Method Comparison:")
        print(
            f"{'Method':<12} {'Memory':<12} {'Avg Latency':<15} "
            f"{'Throughput':<20} {'Size Reduction':<15}"
        )
        print("-" * 85)

        fp32_memory = results.get("FP32", {}).get("memory", 0)
        for method, metrics in results.items():
            size_reduction = (
                f"{(1 - metrics['memory']/fp32_memory)*100:.1f}%"
                if fp32_memory > 0
                else "N/A"
            )
            print(
                f"{method:<12} {format_size(metrics['memory']):<12} "
                f"{metrics['avg_latency']:<15} {metrics['throughput']:<20} "
                f"{size_reduction:<15}"
            )

        return results

    def _benchmark_model(
        self, model, tokenizer, test_texts: list[str]
    ) -> dict[str, Any]:
        """Benchmark a quantized model."""
        # Memory usage
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated()
        else:
            memory = 0

        # Warm up
        inputs = tokenizer(
            test_texts[0], return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)

        # Benchmark
        total_time = 0
        for text in test_texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)

            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=20)
            total_time += time.time() - start_time

        avg_latency = total_time / len(test_texts)
        throughput = len(test_texts) / total_time

        return {
            "memory": memory,
            "avg_latency": f"{avg_latency*1000:.1f}ms",
            "throughput": f"{throughput:.2f} seq/s",
            "total_time": total_time,
        }

    def demonstrate_qlora_optimization(
        self, model_name: str = "gpt2"
    ) -> dict[str, Any]:
        """
        Demonstrate QLoRA (Quantized LoRA) for memory-efficient fine-tuning.

        Args:
            model_name: Model to demonstrate with

        Returns:
            QLoRA optimization metrics
        """
        print(f"\n🎯 Demonstrating QLoRA (INT4 + LoRA) for {model_name}")

        # Load model with INT4 quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )

        # Prepare for LoRA
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"] if "gpt2" in model_name else ["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        # Print statistics
        model.print_trainable_parameters()

        # Calculate memory footprint
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Approximate memory usage
        base_model_memory = total_params * 0.5  # INT4 = 0.5 bytes per param
        lora_memory = trainable_params * 2  # FP16 = 2 bytes per param
        total_memory = base_model_memory + lora_memory

        metrics = {
            "base_model": model_name,
            "quantization": "INT4 (NF4)",
            "lora_rank": 8,
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "trainable_percentage": f"{(trainable_params/total_params)*100:.3f}%",
            "approx_memory": format_size(total_memory),
            "base_model_memory": format_size(base_model_memory),
            "lora_adapter_memory": format_size(lora_memory),
        }

        print("\n✅ QLoRA Configuration:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics

    def export_quantized_model(
        self, model, output_dir: str = "models/quantized"
    ) -> dict[str, Any]:
        """
        Export quantized model for deployment.

        Args:
            model: Quantized model to export
            output_dir: Directory to save model

        Returns:
            Export metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Exporting quantized model to {output_dir}...")

        # Save model
        model.save_pretrained(output_path)

        # Calculate exported size
        total_size = 0
        for file in output_path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

        metrics = {
            "output_dir": str(output_path),
            "exported_size": format_size(total_size),
            "files_created": len(list(output_path.rglob("*"))),
        }

        print("\n✅ Export complete:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics


def demonstrate_advanced_quantization():
    """Demonstrate advanced quantization techniques."""
    print("=" * 80)
    print("🔢 ADVANCED QUANTIZATION DEMONSTRATION")
    print("=" * 80)

    quant = AdvancedQuantization()

    # 1. INT4 Quantization
    print("\n1️⃣ INT4 Quantization")
    int4_metrics, int4_model = quant.quantize_model_int4()

    # 2. Compare quantization methods
    print("\n2️⃣ Quantization Method Comparison")
    comparison = quant.compare_quantization_methods(test_sequences=5)

    # 3. QLoRA demonstration
    print("\n3️⃣ QLoRA (Quantized LoRA)")
    qlora_metrics = quant.demonstrate_qlora_optimization()

    # 4. Export quantized model
    print("\n4️⃣ Export Quantized Model")
    export_metrics = quant.export_quantized_model(int4_model)

    print("\n" + "=" * 80)
    print("📊 QUANTIZATION SUMMARY")
    print("=" * 80)

    print("\n✅ Key Benefits Demonstrated:")
    print("   - INT4: 87.5% size reduction vs FP32")
    print("   - INT8: 75% size reduction with minimal accuracy loss")
    print("   - QLoRA: Fine-tune large models on consumer GPUs")
    print("   - Double quantization: Extra memory savings")

    print("\n💡 Deployment Recommendations:")
    print("   - Edge devices: INT8 dynamic quantization")
    print("   - GPU servers: INT4 with FP16 compute")
    print("   - Fine-tuning: QLoRA for memory efficiency")
    print("   - Mobile: Export to ONNX after quantization")


if __name__ == "__main__":
    demonstrate_advanced_quantization()
