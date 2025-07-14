"""
PEFT (Parameter-Efficient Fine-Tuning) and LoRA integration examples.

This module demonstrates how to use LoRA and other PEFT methods to efficiently
fine-tune large language models with minimal compute and memory requirements.
"""

import os
import time
from typing import Any

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.config import Config
from src.utils import format_size, timer_decorator


class PEFTTrainer:
    """Trainer class for PEFT/LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str = "gpt2",  # Start with smaller model for demo
        task_type: TaskType = TaskType.CAUSAL_LM,
    ):
        """Initialize PEFT trainer with model and task type."""
        self.model_name = model_name
        self.task_type = task_type
        self.device = Config.DEVICE

    def setup_lora_model(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list[str] = None,
    ) -> tuple:
        """
        Setup model with LoRA configuration.

        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        print(f"\n🔧 Setting up LoRA for {self.model_name}...")

        # Load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Print original model stats
        original_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Original model parameters: {original_params:,}")

        # Configure LoRA
        if target_modules is None:
            # Default modules for different model types
            if "gpt2" in self.model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in self.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            else:
                target_modules = ["query", "value"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=self.task_type,
            target_modules=target_modules,
        )

        # Create PEFT model
        model = prepare_model_for_kbit_training(model)
        peft_model = get_peft_model(model, lora_config)

        # Print LoRA stats
        peft_model.print_trainable_parameters()

        return peft_model, tokenizer

    @timer_decorator
    def fine_tune_with_lora(
        self,
        peft_model,
        tokenizer,
        dataset: Dataset,
        output_dir: str = "models/lora",
        num_epochs: int = 3,
        batch_size: int = 4,
    ) -> dict[str, Any]:
        """
        Fine-tune model using LoRA.

        Args:
            peft_model: PEFT model with LoRA
            tokenizer: Tokenizer
            dataset: Training dataset
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics
        """
        print("\n🚀 Starting LoRA fine-tuning...")

        # Prepare dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=128
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            learning_rate=3e-4,
            fp16=self.device.type == "cuda",
            push_to_hub=False,
        )

        # Create trainer
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Save LoRA weights
        peft_model.save_pretrained(output_dir)

        # Calculate adapter size
        adapter_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if f.endswith(".bin") or f.endswith(".safetensors")
        )

        metrics = {
            "training_time": f"{training_time:.2f}s",
            "adapter_size": format_size(adapter_size),
            "trainable_params": sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            ),
            "total_params": sum(p.numel() for p in peft_model.parameters()),
            "efficiency": f"{(sum(p.numel() for p in peft_model.parameters() if p.requires_grad) / sum(p.numel() for p in peft_model.parameters()) * 100):.2f}%",
        }

        print("\n✅ LoRA Training Complete:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        return metrics

    def inference_with_lora(
        self,
        base_model_name: str,
        lora_weights_path: str,
        prompt: str,
        max_length: int = 100,
    ) -> str:
        """
        Run inference with LoRA adapter.

        Args:
            base_model_name: Name of base model
            lora_weights_path: Path to LoRA weights
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Generated text
        """
        print("\n💭 Running inference with LoRA adapter...")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(model, lora_weights_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def compare_lora_configs(self, dataset: Dataset) -> dict[str, Any]:
        """
        Compare different LoRA configurations.

        Args:
            dataset: Training dataset

        Returns:
            Comparison metrics
        """
        print("\n🔬 Comparing LoRA Configurations...")

        configs = [
            {"r": 8, "lora_alpha": 16, "name": "Small (r=8)"},
            {"r": 16, "lora_alpha": 32, "name": "Medium (r=16)"},
            {"r": 32, "lora_alpha": 64, "name": "Large (r=32)"},
        ]

        results = []

        for config in configs:
            print(f"\n📊 Testing {config['name']}...")

            # Setup model with config
            peft_model, tokenizer = self.setup_lora_model(
                r=config["r"], lora_alpha=config["lora_alpha"]
            )

            # Count trainable parameters
            trainable_params = sum(
                p.numel() for p in peft_model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in peft_model.parameters())

            # Measure memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()

                # Do a forward pass
                dummy_input = tokenizer("Test", return_tensors="pt").to(self.device)
                _ = peft_model(**dummy_input)

                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
            else:
                memory_used = 0

            results.append(
                {
                    "config": config["name"],
                    "r": config["r"],
                    "trainable_params": f"{trainable_params:,}",
                    "total_params": f"{total_params:,}",
                    "efficiency": f"{(trainable_params/total_params*100):.3f}%",
                    "memory_used": (
                        format_size(memory_used) if memory_used > 0 else "N/A"
                    ),
                }
            )

            # Clean up
            del peft_model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        print("\n📊 LoRA Configuration Comparison:")
        print(
            f"{'Config':<15} {'r':<5} {'Trainable':<15} {'Total':<15} "
            f"{'Efficiency':<10} {'Memory':<10}"
        )
        print("-" * 80)
        for result in results:
            print(
                f"{result['config']:<15} {result['r']:<5} "
                f"{result['trainable_params']:<15} "
                f"{result['total_params']:<15} {result['efficiency']:<10} "
                f"{result['memory_used']:<10}"
            )

        return {"comparison": results}


def demonstrate_peft_lora():
    """Demonstrate PEFT/LoRA workflow."""
    print("=" * 80)
    print("🎯 PEFT/LoRA DEMONSTRATION")
    print("=" * 80)

    # Create sample dataset
    texts = [
        "The future of AI is bright and full of possibilities.",
        "Machine learning transforms how we solve complex problems.",
        "Deep learning models continue to improve rapidly.",
        "Natural language processing enables better human-computer interaction.",
        "Computer vision applications are becoming more sophisticated.",
    ] * 20  # Repeat for larger dataset

    dataset = Dataset.from_dict({"text": texts})

    # Initialize trainer
    trainer = PEFTTrainer(model_name="gpt2")

    # 1. Setup LoRA model
    print("\n1️⃣ Setting up LoRA Model")
    peft_model, tokenizer = trainer.setup_lora_model(r=16, lora_alpha=32)

    # 2. Fine-tune with LoRA
    print("\n2️⃣ Fine-tuning with LoRA")
    metrics = trainer.fine_tune_with_lora(
        peft_model, tokenizer, dataset, num_epochs=1  # Quick demo
    )

    # 3. Test inference
    print("\n3️⃣ Testing Inference")
    generated = trainer.inference_with_lora(
        "gpt2", "models/lora", "The future of AI", max_length=50
    )
    print(f"Generated: {generated}")

    # 4. Compare configurations
    print("\n4️⃣ Comparing Configurations")
    comparison = trainer.compare_lora_configs(dataset)

    print("\n" + "=" * 80)
    print("✅ PEFT/LoRA demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_peft_lora()
