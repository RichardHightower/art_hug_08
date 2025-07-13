"""
Edge deployment utilities for exporting models to ONNX and other formats.

This module demonstrates how to export HuggingFace models for edge deployment,
including ONNX export, optimization, and quantization for mobile/edge devices.
"""

import os
import time
from pathlib import Path
from typing import Any

import torch
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config
from utils import format_size, timer_decorator


class EdgeDeploymentPipeline:
    """Pipeline for exporting and optimizing models for edge deployment."""

    def __init__(self, model_name: str = None):
        """Initialize with a model for edge deployment."""
        self.model_name = model_name or Config.DEFAULT_SENTIMENT_MODEL
        self.device = torch.device("cpu")  # Edge deployment typically on CPU

    def export_to_onnx(
        self, output_dir: str = "models/onnx", optimize: bool = True
    ) -> dict[str, Any]:
        """
        Export model to ONNX format with optimization.

        Args:
            output_dir: Directory to save ONNX model
            optimize: Whether to apply ONNX optimizations

        Returns:
            Dictionary with export metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n🔧 Exporting {self.model_name} to ONNX...")

        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Get model size before export
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Create dummy input for tracing
        dummy_text = "This is a sample text for ONNX export"
        inputs = tokenizer(dummy_text, return_tensors="pt")

        # Export to ONNX
        start_time = time.time()

        onnx_path = output_path / "model.onnx"
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
        )

        export_time = time.time() - start_time

        # Optimize if requested
        if optimize:
            print("📊 Applying ONNX optimizations...")
            optimizer = ORTOptimizer.from_pretrained(self.model_name)

            optimization_config = OptimizationConfig(
                optimization_level=2,
                optimize_for_gpu=False,
                fp16=False,  # Keep FP32 for CPU
                enable_transformers_specific_optimizations=True,
            )

            optimizer.optimize(
                optimization_config=optimization_config, save_dir=output_path
            )

        # Get final size
        onnx_size = os.path.getsize(onnx_path)

        metrics = {
            "original_size": format_size(original_size),
            "onnx_size": format_size(onnx_size),
            "size_reduction": f"{(1 - onnx_size/original_size)*100:.1f}%",
            "export_time": f"{export_time:.2f}s",
            "output_path": str(onnx_path),
        }

        print("\n✅ ONNX Export Complete:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics

    def export_for_mobile(
        self, output_dir: str = "models/mobile", quantize: bool = True
    ) -> dict[str, Any]:
        """
        Export model optimized for mobile deployment.

        Args:
            output_dir: Directory to save mobile-optimized model
            quantize: Whether to apply dynamic quantization

        Returns:
            Dictionary with mobile optimization metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📱 Optimizing {self.model_name} for mobile...")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()

        # Apply dynamic quantization
        if quantize:
            print("🔢 Applying dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            quantized_model = model

        # Save the model
        torch.save(quantized_model, output_path / "mobile_model.pt")

        # Also save in TorchScript format for mobile
        print("📝 Creating TorchScript model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dummy_input = tokenizer("Sample text", return_tensors="pt")

        traced_model = torch.jit.trace(
            quantized_model, (dummy_input["input_ids"], dummy_input["attention_mask"])
        )

        traced_model.save(output_path / "mobile_model_traced.pt")

        # Calculate sizes
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        mobile_size = os.path.getsize(output_path / "mobile_model.pt")
        traced_size = os.path.getsize(output_path / "mobile_model_traced.pt")

        metrics = {
            "original_size": format_size(original_size),
            "mobile_size": format_size(mobile_size),
            "traced_size": format_size(traced_size),
            "size_reduction": f"{(1 - mobile_size/original_size)*100:.1f}%",
            "quantization": "INT8" if quantize else "FP32",
        }

        print("\n✅ Mobile Optimization Complete:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics

    @timer_decorator
    def benchmark_edge_inference(
        self, model_path: str, num_samples: int = 100
    ) -> dict[str, Any]:
        """
        Benchmark inference performance on edge device.

        Args:
            model_path: Path to the optimized model
            num_samples: Number of samples to benchmark

        Returns:
            Performance metrics
        """
        print(f"\n⚡ Benchmarking edge inference with {num_samples} samples...")

        # Load optimized model
        if model_path.endswith(".onnx"):
            # ONNX Runtime inference
            import onnxruntime as ort

            session = ort.InferenceSession(model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Warm up
            dummy_input = tokenizer("Warm up", return_tensors="np")
            _ = session.run(
                None,
                {
                    "input_ids": dummy_input["input_ids"],
                    "attention_mask": dummy_input["attention_mask"],
                },
            )

            # Benchmark
            texts = [f"Sample text {i}" for i in range(num_samples)]
            start_time = time.time()

            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                _ = session.run(
                    None,
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    },
                )

            total_time = time.time() - start_time

        else:
            # PyTorch inference
            model = torch.load(model_path)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Warm up
            with torch.no_grad():
                dummy_input = tokenizer("Warm up", return_tensors="pt")
                _ = model(**dummy_input)

            # Benchmark
            texts = [f"Sample text {i}" for i in range(num_samples)]
            start_time = time.time()

            with torch.no_grad():
                for text in texts:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    )
                    _ = model(**inputs)

            total_time = time.time() - start_time

        avg_latency = (total_time / num_samples) * 1000  # Convert to ms
        throughput = num_samples / total_time

        metrics = {
            "total_samples": num_samples,
            "total_time": f"{total_time:.2f}s",
            "avg_latency": f"{avg_latency:.2f}ms",
            "throughput": f"{throughput:.1f} samples/sec",
            "model_type": "ONNX" if model_path.endswith(".onnx") else "PyTorch",
        }

        print("\n📊 Edge Inference Benchmarks:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics


def demonstrate_edge_deployment():
    """Demonstrate complete edge deployment workflow."""
    print("=" * 80)
    print("🚀 EDGE DEPLOYMENT DEMONSTRATION")
    print("=" * 80)

    pipeline = EdgeDeploymentPipeline()

    # 1. Export to ONNX
    print("\n1️⃣ ONNX Export")
    onnx_metrics = pipeline.export_to_onnx()

    # 2. Mobile optimization
    print("\n2️⃣ Mobile Optimization")
    mobile_metrics = pipeline.export_for_mobile()

    # 3. Benchmark performance
    print("\n3️⃣ Performance Benchmarking")

    # Benchmark ONNX
    onnx_perf = pipeline.benchmark_edge_inference(
        "models/onnx/model.onnx", num_samples=50
    )

    # Benchmark mobile model
    mobile_perf = pipeline.benchmark_edge_inference(
        "models/mobile/mobile_model.pt", num_samples=50
    )

    print("\n" + "=" * 80)
    print("📈 DEPLOYMENT SUMMARY")
    print("=" * 80)
    print("\nONNX Model:")
    print(
        f"  Size: {onnx_metrics['onnx_size']} ({onnx_metrics['size_reduction']} smaller)"
    )
    print(f"  Latency: {onnx_perf['avg_latency']}")
    print(f"  Throughput: {onnx_perf['throughput']}")

    print("\nMobile Model:")
    print(
        f"  Size: {mobile_metrics['mobile_size']} ({mobile_metrics['size_reduction']} smaller)"
    )
    print(f"  Latency: {mobile_perf['avg_latency']}")
    print(f"  Throughput: {mobile_perf['throughput']}")


if __name__ == "__main__":
    demonstrate_edge_deployment()
