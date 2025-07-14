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
    BitsAndBytesConfig,
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
        model_name: str = "microsoft/phi-2",  # Use modern model
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
        use_qlora: bool = False,
    ) -> tuple:
        """
        Setup model with LoRA configuration.

        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to
            use_qlora: Use QLoRA (INT4 quantization) for memory efficiency

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        print(f"\n🔧 Setting up {'QLoRA' if use_qlora else 'LoRA'} for {self.model_name}...")

        # Configure quantization for QLoRA
        if use_qlora:
            print("⚙️  Configuring INT4 quantization for QLoRA...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # Load base model without quantization
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
            elif "phi" in self.model_name.lower():
                target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
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

    # Initialize trainer with modern model
    model_name = "microsoft/phi-2"
    trainer = PEFTTrainer(model_name=model_name)

    # 1. Setup standard LoRA model
    print("\n1️⃣ Setting up Standard LoRA Model")
    peft_model, tokenizer = trainer.setup_lora_model(r=16, lora_alpha=32, use_qlora=False)

    # 2. Setup QLoRA model (INT4 quantized)
    print("\n2️⃣ Setting up QLoRA Model (INT4 Quantized)")
    qlora_trainer = PEFTTrainer(model_name=model_name)
    qlora_model, qlora_tokenizer = qlora_trainer.setup_lora_model(
        r=16, lora_alpha=32, use_qlora=True
    )

    # 3. Fine-tune with standard LoRA
    print("\n3️⃣ Fine-tuning with Standard LoRA")
    metrics = trainer.fine_tune_with_lora(
        peft_model, tokenizer, dataset, num_epochs=1  # Quick demo
    )

    # 4. Fine-tune with QLoRA
    print("\n4️⃣ Fine-tuning with QLoRA")
    qlora_metrics = qlora_trainer.fine_tune_with_lora(
        qlora_model, qlora_tokenizer, dataset, 
        output_dir="models/qlora",
        num_epochs=1  # Quick demo
    )

    # 5. Test inference
    print("\n5️⃣ Testing Inference")
    generated = trainer.inference_with_lora(
        model_name, "models/lora", "The future of AI", max_length=50
    )
    print(f"Generated (LoRA): {generated}")
    
    qlora_generated = qlora_trainer.inference_with_lora(
        model_name, "models/qlora", "The future of AI", max_length=50
    )
    print(f"Generated (QLoRA): {qlora_generated}")

    # 6. Compare configurations
    print("\n6️⃣ Comparing Configurations")
    comparison = trainer.compare_lora_configs(dataset)
    
    # 7. Summary comparison
    print("\n7️⃣ LoRA vs QLoRA Comparison")
    print(f"{'Method':<15} {'Adapter Size':<15} {'Training Time':<15} {'Memory Usage':<15}")
    print("-" * 60)
    print(f"{'Standard LoRA':<15} {metrics['adapter_size']:<15} {metrics['training_time']:<15} {'High':<15}")
    print(f"{'QLoRA (INT4)':<15} {qlora_metrics['adapter_size']:<15} {qlora_metrics['training_time']:<15} {'Low (~25%)':<15}")

    print("\n" + "=" * 80)
    print("✅ PEFT/LoRA demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PEFT/LoRA demonstration")
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Run QLoRA demonstration only"
    )
    args = parser.parse_args()
    
    if args.qlora:
        # Run QLoRA-specific demo
        print("=" * 80)
        print("🎯 QLORA DEMONSTRATION")
        print("=" * 80)
        
        # Create sample dataset
        texts = [
            "The future of AI is bright and full of possibilities.",
            "Machine learning transforms how we solve complex problems.",
            "Deep learning models continue to improve rapidly.",
        ] * 10
        dataset = Dataset.from_dict({"text": texts})
        
        # Run QLoRA demo
        trainer = PEFTTrainer(model_name="microsoft/phi-2")
        qlora_model, tokenizer = trainer.setup_lora_model(
            r=8, lora_alpha=16, use_qlora=True
        )
        
        print("\n✅ QLoRA setup complete!")
        print("   - INT4 quantization enabled")
        print("   - Memory usage reduced by ~75%")
        print("   - Ready for fine-tuning large models on consumer GPUs")
    else:
        demonstrate_peft_lora()
