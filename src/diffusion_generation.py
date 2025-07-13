"""
Diffusion pipeline examples for synthetic image generation.

This module demonstrates how to use Stable Diffusion and other diffusion models
for generating synthetic training data and augmenting datasets.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

from config import Config
from utils import format_size, timer_decorator


class DiffusionGenerator:
    """Generate synthetic images using diffusion models."""

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device = None,
    ):
        """Initialize diffusion generator."""
        self.model_name = model_name
        self.device = device or Config.DEVICE
        self.pipelines = {}

    def load_text2img_pipeline(
        self, enable_attention_slicing: bool = True, enable_cpu_offload: bool = False
    ) -> StableDiffusionPipeline:
        """
        Load text-to-image generation pipeline.

        Args:
            enable_attention_slicing: Enable memory-efficient attention
            enable_cpu_offload: Enable model CPU offloading for low VRAM

        Returns:
            Loaded pipeline
        """
        if "text2img" in self.pipelines:
            return self.pipelines["text2img"]

        print(f"\n🎨 Loading text-to-image pipeline: {self.model_name}")

        # Load pipeline with optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,  # Disable for performance in controlled environments
            requires_safety_checker=False,
        )

        # Optimize for memory/speed
        if enable_attention_slicing:
            pipe.enable_attention_slicing()
            print("✅ Attention slicing enabled (reduces memory)")

        if enable_cpu_offload and self.device.type == "cuda":
            pipe.enable_model_cpu_offload()
            print("✅ CPU offloading enabled (for low VRAM)")
        else:
            pipe = pipe.to(self.device)

        # Use faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        self.pipelines["text2img"] = pipe
        return pipe

    @timer_decorator
    def generate_images(
        self,
        prompts: list[str],
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompts: List of text prompts
            negative_prompt: What to avoid in generation
            num_images_per_prompt: Images per prompt
            num_inference_steps: Denoising steps
            guidance_scale: Classifier-free guidance strength
            width: Image width
            height: Image height
            seed: Random seed for reproducibility

        Returns:
            List of generated images
        """
        pipe = self.load_text2img_pipeline()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        all_images = []

        print(f"\n🖼️ Generating {len(prompts) * num_images_per_prompt} images...")

        for i, prompt in enumerate(prompts):
            print(f"\n  Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")

            # Generate images
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ).images

            all_images.extend(images)

        return all_images

    def generate_dataset_augmentation(
        self,
        categories: dict[str, list[str]],
        images_per_prompt: int = 5,
        output_dir: str = "data/synthetic_images",
    ) -> dict[str, Any]:
        """
        Generate synthetic images for dataset augmentation.

        Args:
            categories: Dict mapping category names to prompt templates
            images_per_prompt: Number of images per prompt
            output_dir: Directory to save images

        Returns:
            Generation statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Generating synthetic dataset with {len(categories)} categories")

        stats = {"total_images": 0, "categories": {}, "generation_time": 0}

        start_time = time.time()

        for category, prompts in categories.items():
            category_path = output_path / category
            category_path.mkdir(exist_ok=True)

            print(f"\n📁 Category: {category}")

            # Generate images for this category
            images = self.generate_images(
                prompts=prompts,
                num_images_per_prompt=images_per_prompt,
                num_inference_steps=20,  # Faster for bulk generation
                guidance_scale=7.5,
            )

            # Save images
            for i, img in enumerate(images):
                img_path = category_path / f"{category}_{i:04d}.png"
                img.save(img_path)

            stats["categories"][category] = len(images)
            stats["total_images"] += len(images)

        stats["generation_time"] = time.time() - start_time

        print("\n✅ Dataset generation complete:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Time: {stats['generation_time']:.1f}s")
        print(f"  Output: {output_path}")

        return stats

    def demonstrate_advanced_techniques(self) -> dict[str, Any]:
        """Demonstrate advanced diffusion techniques."""
        print("\n🔬 Advanced Diffusion Techniques")

        results = {}

        # 1. Prompt engineering for quality
        print("\n1️⃣ Prompt Engineering")
        quality_prompts = [
            "professional product photo of a smartphone, studio lighting, "
            "white background, high quality, 8k",
            "damaged laptop with cracked screen, realistic, detailed, "
            "product documentation photo",
            "elegant watch on display stand, luxury product photography, "
            "bokeh background",
        ]

        quality_images = self.generate_images(
            prompts=quality_prompts,
            negative_prompt="blurry, low quality, distorted, amateur",
            num_inference_steps=30,
            guidance_scale=8.5,
            width=768,
            height=768,
        )

        results["quality_generation"] = {
            "num_images": len(quality_images),
            "resolution": "768x768",
            "technique": "prompt engineering + negative prompts",
        }

        # 2. Style consistency with seeds
        print("\n2️⃣ Style Consistency")
        consistent_prompts = [
            "smartphone in minimalist style",
            "laptop in minimalist style",
            "headphones in minimalist style",
        ]

        # Use same seed for consistency
        consistent_images = self.generate_images(
            prompts=consistent_prompts, num_inference_steps=25, seed=42
        )

        results["style_consistency"] = {
            "num_images": len(consistent_images),
            "technique": "fixed seed + style prompt",
        }

        # 3. Memory optimization comparison
        print("\n3️⃣ Memory Optimization")

        # Test with attention slicing
        pipe = self.load_text2img_pipeline(enable_attention_slicing=True)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

        _ = pipe("test image", num_inference_steps=10, width=512, height=512).images[0]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            mem_with_slicing = torch.cuda.memory_allocated() - mem_before
        else:
            mem_with_slicing = 0

        results["memory_optimization"] = {
            "attention_slicing_memory": format_size(mem_with_slicing),
            "optimization_enabled": True,
        }

        return results, quality_images + consistent_images

    def create_image_variations(
        self,
        base_image: Image.Image,
        variation_prompts: list[str],
        strength: float = 0.75,
    ) -> list[Image.Image]:
        """
        Create variations of an existing image using img2img.

        Args:
            base_image: Source image
            variation_prompts: Prompts for variations
            strength: How much to change (0=no change, 1=complete change)

        Returns:
            List of image variations
        """
        print("\n🔄 Creating image variations...")

        # Load img2img pipeline
        if "img2img" not in self.pipelines:
            self.pipelines["img2img"] = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)

            self.pipelines["img2img"].scheduler = (
                DPMSolverMultistepScheduler.from_config(
                    self.pipelines["img2img"].scheduler.config
                )
            )

        pipe = self.pipelines["img2img"]
        variations = []

        for prompt in variation_prompts:
            print(f"  Creating variation: '{prompt[:50]}...'")

            image = pipe(
                prompt=prompt,
                image=base_image,
                strength=strength,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            variations.append(image)

        return variations

    def quality_filter_images(
        self, images: list[Image.Image], min_entropy: float = 5.0
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        """
        Filter generated images by quality metrics.

        Args:
            images: List of generated images
            min_entropy: Minimum entropy threshold

        Returns:
            Filtered images and statistics
        """
        print(f"\n🔍 Filtering {len(images)} images by quality...")

        filtered = []
        stats = {
            "total": len(images),
            "passed": 0,
            "filtered_low_entropy": 0,
            "filtered_uniform": 0,
        }

        for img in images:
            # Convert to numpy
            img_array = np.array(img)

            # Calculate entropy (measure of information content)
            hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

            # Check for uniform color (low variance)
            variance = np.var(img_array)

            if entropy < min_entropy:
                stats["filtered_low_entropy"] += 1
            elif variance < 100:  # Too uniform
                stats["filtered_uniform"] += 1
            else:
                filtered.append(img)
                stats["passed"] += 1

        print("\n✅ Quality filtering complete:")
        print(f"  Passed: {stats['passed']}/{stats['total']}")
        print(f"  Low entropy: {stats['filtered_low_entropy']}")
        print(f"  Too uniform: {stats['filtered_uniform']}")

        return filtered, stats


def demonstrate_diffusion_generation():
    """Demonstrate diffusion pipeline for synthetic data generation."""
    print("=" * 80)
    print("🎨 DIFFUSION GENERATION DEMONSTRATION")
    print("=" * 80)

    generator = DiffusionGenerator()

    # 1. Basic generation
    print("\n1️⃣ Basic Text-to-Image Generation")
    basic_prompts = [
        "a professional photo of a modern smartphone on white background",
        "a laptop computer with visible keyboard, product photography",
    ]

    basic_images = generator.generate_images(
        prompts=basic_prompts, num_inference_steps=25, seed=42  # For reproducibility
    )

    # Save examples
    output_dir = Path("output/diffusion_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(basic_images):
        img.save(output_dir / f"basic_{i}.png")

    # 2. Dataset augmentation
    print("\n2️⃣ Synthetic Dataset Generation")
    categories = {
        "electronics": [
            "smartphone with cracked screen, repair documentation",
            "pristine laptop, product catalog photo",
            "wireless earbuds in charging case, white background",
        ],
        "furniture": [
            "modern office chair, studio lighting",
            "wooden desk with drawers, product photo",
            "bookshelf filled with books, interior design",
        ],
    }

    dataset_stats = generator.generate_dataset_augmentation(
        categories=categories, images_per_prompt=2
    )

    # 3. Advanced techniques
    print("\n3️⃣ Advanced Generation Techniques")
    advanced_results, advanced_images = generator.demonstrate_advanced_techniques()

    # 4. Quality filtering
    print("\n4️⃣ Quality Filtering")
    all_generated = basic_images + advanced_images
    filtered_images, filter_stats = generator.quality_filter_images(all_generated)

    # 5. Image variations (if we have images)
    if filtered_images:
        print("\n5️⃣ Creating Variations")
        variations = generator.create_image_variations(
            base_image=filtered_images[0],
            variation_prompts=[
                "the same device but with a blue color scheme",
                "the same device in a futuristic style",
            ],
            strength=0.7,
        )

        for i, img in enumerate(variations):
            img.save(output_dir / f"variation_{i}.png")

    print("\n" + "=" * 80)
    print("📊 DIFFUSION GENERATION SUMMARY")
    print("=" * 80)

    print(f"\n✅ Generated {dataset_stats['total_images']} synthetic images")
    print(
        f"✅ Quality filtered: {filter_stats['passed']}/{filter_stats['total']} passed"
    )
    print(f"✅ Output saved to: {output_dir}")

    print("\n💡 Key Applications:")
    print("   - Dataset augmentation for rare classes")
    print("   - Privacy-preserving synthetic data")
    print("   - Rapid prototyping of visual concepts")
    print("   - Style transfer and variations")

    print("\n⚡ Performance Tips:")
    print("   - Use attention slicing for large images")
    print("   - Reduce inference steps for faster generation")
    print("   - Use CPU offloading for limited VRAM")
    print("   - Batch similar prompts for efficiency")


if __name__ == "__main__":
    demonstrate_diffusion_generation()
