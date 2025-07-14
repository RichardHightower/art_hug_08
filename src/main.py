"""Main entry point for workflow demonstrations."""

import argparse

from src.advanced_quantization import demonstrate_advanced_quantization
from src.custom_pipelines import demonstrate_custom_pipelines
from src.data_workflows import demonstrate_data_workflows
from src.diffusion_generation import demonstrate_diffusion_generation
from src.edge_deployment import demonstrate_edge_deployment
from src.flash_attention import demonstrate_flash_attention
from src.optimization import demonstrate_optimization
from src.peft_lora import demonstrate_peft_lora
from src.production_workflows import demonstrate_production_workflow
from src.synthetic_data import demonstrate_synthetic_data


def main():
    """Run all workflow demonstrations."""
    parser = argparse.ArgumentParser(
        description="Custom Pipelines and Data Workflows Demonstrations"
    )
    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "pipelines",
            "data",
            "optimization",
            "synthetic",
            "production",
            "edge",
            "peft",
            "flash",
            "quantization",
            "diffusion",
        ],
        default="all",
        help="Which demonstration to run",
    )
    args = parser.parse_args()

    print("=== Hugging Face Workflow Mastery ===\n")

    if args.demo in ["all", "pipelines"]:
        print("\n--- Custom Pipeline Demonstrations ---")
        demonstrate_custom_pipelines()

    if args.demo in ["all", "data"]:
        print("\n--- Data Workflow Demonstrations ---")
        demonstrate_data_workflows()

    if args.demo in ["all", "optimization"]:
        print("\n--- Optimization Demonstrations ---")
        demonstrate_optimization()

    if args.demo in ["all", "synthetic"]:
        print("\n--- Synthetic Data Demonstrations ---")
        demonstrate_synthetic_data()

    if args.demo in ["all", "production"]:
        print("\n--- Production Workflow Demonstration ---")
        demonstrate_production_workflow()

    if args.demo in ["all", "edge"]:
        print("\n--- Edge Deployment Demonstration ---")
        demonstrate_edge_deployment()

    try:
        if args.demo in ["all", "peft"]:
            print("\n--- PEFT/LoRA Demonstration ---")
            demonstrate_peft_lora()
    except Exception as e:
        print(f"Error running PEFT/LoRA demonstration: {e} You need CUDA")

    if args.demo in ["all", "flash"]:
        print("\n--- Flash Attention Demonstration ---")
        demonstrate_flash_attention()


    try:
        if args.demo in ["all", "quantization"]:
            print("\n--- Advanced Quantization Demonstration ---")
            demonstrate_advanced_quantization()
    except Exception as e:
        print(f"Error running Advanced Quantization Demonstration: {e} \nYou need CUDA")

    if args.demo in ["all", "diffusion"]:
        print("\n--- Diffusion Generation Demonstration ---")
        demonstrate_diffusion_generation()

    print("\nAll demonstrations complete!")


if __name__ == "__main__":
    main()
