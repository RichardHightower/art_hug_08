# Customizing Pipelines and Data Workflows: Advanced Models and Efficient Processing

This project contains working examples for Chapter 8 of the Hugging Face Transformers book, demonstrating how to transform from pipeline user to workflow architect.

## Overview

Learn how to implement and understand:

- Pipeline anatomy and customization
- Component swapping and composition
- Efficient data handling with 🤗 Datasets
- Streaming for massive datasets
- Model optimization (quantization, batching, edge deployment)
- Synthetic data generation with LLMs and diffusion models
- Production-ready workflows with cost optimization

## Prerequisites

- Python 3.12 (managed via pyenv)
- Poetry for dependency management
- Go Task for build automation
- CUDA-capable GPU recommended (but CPU mode supported)
- (Optional) Hugging Face account for accessing gated models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd huggingface-workflows
```

2. Set up Python environment:
```bash
pyenv install 3.12.9
pyenv local 3.12.9
```

3. Install dependencies:
```bash
poetry install
poetry shell
```

4. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

Run the main demonstration:
```bash
task run
```

Or use specific modules:
```bash
# Custom pipeline examples
python -m src.custom_pipelines

# Data workflow demonstrations
python -m src.data_workflows

# Optimization benchmarks
python -m src.optimization

# Synthetic data generation
python -m src.synthetic_data
```

## Project Structure

```
huggingface-workflows/
├── src/
│   ├── custom_pipelines.py     # Pipeline customization examples
│   ├── data_workflows.py        # Efficient data handling
│   ├── optimization.py          # Model optimization techniques
│   ├── synthetic_data.py        # Data generation methods
│   ├── production_workflows.py  # End-to-end workflows
│   ├── edge_deployment.py       # ONNX export and edge deployment
│   ├── peft_lora.py            # PEFT/LoRA fine-tuning examples
│   ├── flash_attention.py       # Flash Attention demonstrations
│   ├── advanced_quantization.py # INT4/INT8 quantization
│   ├── diffusion_generation.py  # Stable Diffusion for images
│   └── utils.py                # Helper functions
├── tests/
│   └── test_workflows.py      # Unit tests
├── notebooks/
│   ├── pipeline_exploration.ipynb
│   └── optimization_benchmarks.ipynb
└── examples/
    └── retail_workflow.py     # Real-world retail example
```

## Key Features

### 1. Custom Pipeline Creation
- Subclass and extend standard pipelines
- Chain multiple models together
- Add business logic and preprocessing

### 2. Efficient Data Processing
- Stream datasets without memory limits
- Parallel transformations with `map()`
- Smart batching for 10x speedup

### 3. Model Optimization
- INT8/INT4 quantization for 75% cost reduction
- Edge deployment strategies
- PEFT/LoRA for efficient fine-tuning

### 4. Synthetic Data Generation
- LLM-based text generation
- SDXL image creation
- Quality validation pipelines

## Running Tests

```bash
task test
```

## Available Tasks

See all available tasks:
```bash
task --list
```

## Learning Path

1. Start with `custom_pipelines.py` to understand pipeline anatomy
2. Explore `data_workflows.py` for handling large-scale data
3. Run `optimization.py` to see performance improvements
4. Experiment with `synthetic_data.py` for data augmentation
5. Study `production_workflows.py` for real-world patterns

## Performance Benchmarks

| Technique | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Batching | 50ms/item | 6.25ms/item | 8x faster |
| INT8 Quantization | 400MB | 100MB | 75% smaller |
| Streaming | 100GB RAM | 200MB RAM | 500x less |
| PEFT Fine-tuning | 13GB | 40MB | 99.7% fewer params |

## Resources

- [Hugging Face Pipelines Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Datasets Library Guide](https://huggingface.co/docs/datasets)
- [Quantization Tutorial](https://huggingface.co/docs/transformers/quantization)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

This project is licensed under the MIT License.
