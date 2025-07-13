"""Synthetic data generation for training augmentation."""

import json
import random
from collections import Counter

import numpy as np
import torch
from transformers import pipeline

from src.config import SYNTHETIC_DATA_PATH, VALIDATION_THRESHOLD


def generate_text_variations(base_prompt, model="gpt2", num_samples=5):
    """Generate text variations using language models."""
    generator = pipeline(
        "text-generation", model=model, device=0 if torch.cuda.is_available() else -1
    )

    variations = []
    temperatures = [0.7, 0.8, 0.9, 1.0, 1.1]

    for i in range(num_samples):
        temp = temperatures[i % len(temperatures)]

        result = generator(
            base_prompt,
            max_new_tokens=50,
            temperature=temp,
            num_return_sequences=1,
            pad_token_id=50256,  # GPT2 eos token
            do_sample=True,
        )[0]["generated_text"]

        # Extract only the generated part
        if base_prompt in result:
            generated = result[len(base_prompt) :].strip()
        else:
            generated = result

        variations.append(
            {
                "prompt": base_prompt,
                "generated": generated,
                "temperature": temp,
                "full_text": result,
            }
        )

    return variations


def validate_synthetic_text(synthetic_samples, real_samples):
    """Validate synthetic text quality."""
    # Calculate basic statistics
    real_lengths = [len(s.split()) for s in real_samples]
    synth_lengths = [len(s["full_text"].split()) for s in synthetic_samples]

    real_mean = np.mean(real_lengths)
    synth_mean = np.mean(synth_lengths)

    # Length similarity
    length_similarity = 1 - abs(real_mean - synth_mean) / real_mean

    # Vocabulary overlap
    real_words = set(" ".join(real_samples).lower().split())
    synth_words = set(
        " ".join([s["full_text"] for s in synthetic_samples]).lower().split()
    )
    vocab_overlap = len(real_words & synth_words) / len(real_words)

    # Diversity check
    synth_texts = [s["full_text"] for s in synthetic_samples]
    diversity = len(set(synth_texts)) / len(synth_texts)

    validation_score = (length_similarity + vocab_overlap + diversity) / 3

    return {
        "score": validation_score,
        "length_similarity": length_similarity,
        "vocab_overlap": vocab_overlap,
        "diversity": diversity,
        "passed": validation_score >= VALIDATION_THRESHOLD,
    }


def generate_classification_data(categories, samples_per_category=10):
    """Generate synthetic classification data."""
    prompts = {
        "positive": [
            "Write a positive product review about",
            "Express satisfaction with",
            "Describe why you love",
        ],
        "negative": [
            "Write a negative review about",
            "Express disappointment with",
            "Explain problems with",
        ],
        "neutral": [
            "Write a balanced review of",
            "Give a fair assessment of",
            "Describe the pros and cons of",
        ],
    }

    products = [
        "wireless headphones",
        "a smartphone",
        "a laptop",
        "running shoes",
        "a coffee maker",
        "a backpack",
    ]

    synthetic_data = []

    for category in categories:
        if category not in prompts:
            continue

        for _ in range(samples_per_category):
            prompt_template = random.choice(prompts[category])
            product = random.choice(products)
            full_prompt = f"{prompt_template} {product}: "

            variations = generate_text_variations(full_prompt, num_samples=1)

            if variations:
                synthetic_data.append(
                    {
                        "text": variations[0]["full_text"],
                        "label": category,
                        "is_synthetic": True,
                        "prompt_used": full_prompt,
                    }
                )

    return synthetic_data


def demonstrate_synthetic_data():
    """Demonstrate synthetic data generation techniques."""

    print("1. Text Generation with GPT-2")
    print("-" * 40)

    # Generate product reviews
    review_prompts = [
        "This smartphone has amazing",
        "The battery life on this device",
        "I've been using this laptop for",
        "The customer service was",
    ]

    all_variations = []
    for prompt in review_prompts:
        print(f"\nPrompt: '{prompt}'")
        variations = generate_text_variations(prompt, num_samples=3)
        all_variations.extend(variations)

        for i, var in enumerate(variations):
            print(f"  Variation {i+1} (temp={var['temperature']}):")
            print(f"    {var['generated'][:100]}...")

    print("\n2. Synthetic Data Validation")
    print("-" * 40)

    # Mock real samples for comparison
    real_samples = [
        "This smartphone has amazing features and great battery life.",
        "The battery life on this device lasts all day with heavy use.",
        "I've been using this laptop for work and it's been reliable.",
        "The customer service was helpful and resolved my issue quickly.",
    ]

    validation_results = validate_synthetic_text(all_variations, real_samples)

    print("Validation Results:")
    print(f"  Overall Score: {validation_results['score']:.3f}")
    print(f"  Length Similarity: {validation_results['length_similarity']:.3f}")
    print(f"  Vocabulary Overlap: {validation_results['vocab_overlap']:.3f}")
    print(f"  Diversity: {validation_results['diversity']:.3f}")
    print(f"  Passed: {'✓' if validation_results['passed'] else '✗'}")

    print("\n3. Classification Data Generation")
    print("-" * 40)

    synthetic_classification = generate_classification_data(
        ["positive", "negative", "neutral"], samples_per_category=3
    )

    print(f"Generated {len(synthetic_classification)} classification samples:")
    for sample in synthetic_classification[:5]:
        print(f"\nLabel: {sample['label']}")
        print(f"Text: {sample['text'][:150]}...")

    print("\n4. Data Augmentation Strategy")
    print("-" * 40)

    # Simulate class imbalance
    original_distribution = {"positive": 1000, "negative": 200, "neutral": 300}

    print("Original distribution:")
    for label, count in original_distribution.items():
        print(f"  {label}: {count} samples")

    # Calculate how many synthetic samples needed
    max_count = max(original_distribution.values())
    augmentation_needed = {}

    for label, count in original_distribution.items():
        if count < max_count * 0.8:  # Augment if less than 80% of max
            needed = int(max_count * 0.8 - count)
            augmentation_needed[label] = needed

    print("\nAugmentation plan:")
    for label, needed in augmentation_needed.items():
        print(f"  Generate {needed} synthetic {label} samples")

    print("\n5. Quality Control Pipeline")
    print("-" * 40)

    def quality_filter(samples):
        """Filter out low-quality synthetic samples."""
        filtered = []
        reasons = Counter()

        for sample in samples:
            text = sample["text"]

            # Length check
            if len(text.split()) < 5:
                reasons["too_short"] += 1
                continue

            # Repetition check
            words = text.lower().split()
            if len(words) > 0 and len(set(words)) / len(words) < 0.5:
                reasons["repetitive"] += 1
                continue

            # Truncation check
            if text.endswith("...") or text.count("...") > 2:
                reasons["truncated"] += 1
                continue

            filtered.append(sample)

        return filtered, reasons

    filtered_samples, filter_reasons = quality_filter(synthetic_classification)

    print("Quality filtering results:")
    print(f"  Input samples: {len(synthetic_classification)}")
    print(f"  Output samples: {len(filtered_samples)}")
    print(f"  Filtered out: {len(synthetic_classification) - len(filtered_samples)}")

    if filter_reasons:
        print("  Reasons:")
        for reason, count in filter_reasons.items():
            print(f"    {reason}: {count}")

    # Save synthetic data
    output_file = SYNTHETIC_DATA_PATH / "synthetic_samples.json"
    with open(output_file, "w") as f:
        json.dump(filtered_samples, f, indent=2)

    print(f"\nSaved {len(filtered_samples)} samples to {output_file}")


if __name__ == "__main__":
    demonstrate_synthetic_data()
