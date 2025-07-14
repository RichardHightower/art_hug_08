# Quick Start Guide

Welcome to the HuggingFace Workflows project! This guide will get you up and running in minutes.

## 1. Prerequisites

Make sure you have:
- Git
- Task (go-task)
- pyenv

## 2. Clone and Setup

```bash
# Clone the repository
git clone git@github.com:RichardHightower/art_hug_08.git
cd art_hug_08

# Run the complete setup (installs Python, Poetry, and all dependencies)
task setup
```

## 3. Verify Installation

```bash
# Test the environment
task test-env

# Run a quick demo
task quick-start
```

## 4. Explore Examples

### Interactive Tutorial (Recommended)
```bash
poetry run jupyter notebook notebooks/tutorial.ipynb
```

### Individual Demos
```bash
# Custom pipelines
poetry run python src/custom_pipelines.py

# Efficient data handling
poetry run python src/efficient_data_handling.py

# Optimization techniques
poetry run python src/optimization_demo.py

# Production workflow
poetry run python src/production_workflows.py
```

## 5. Advanced Examples

```bash
# PEFT/LoRA fine-tuning
task peft

# Flash Attention
task flash

# Quantization
task quantization

# Diffusion models
task diffusion
```

## Troubleshooting

- **Import errors**: Make sure you're using Poetry: `poetry run python ...`
- **GPU issues**: The code automatically detects and uses available hardware (CUDA/MPS/CPU)
- **Memory errors**: Reduce batch sizes in the examples

## Next Steps

1. Read the improved documentation: `docs/art_08i.md`
2. Work through the tutorial notebook
3. Experiment with the examples
4. Build your own custom workflows!

Happy learning! 🚀