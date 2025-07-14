# Setup Notes - art_hug_08 Project

## Environment Setup Completed Successfully ✅

The project has been successfully set up with the following configuration:

### System Information
- Platform: macOS (Darwin) on Apple Silicon (ARM64)
- Python Version: 3.12.9 (via pyenv)
- Virtual Environment: Poetry with .venv in project directory

### Installed Packages
- ✅ **transformers** ^4.53.0 - Core Hugging Face library
- ✅ **torch** 2.2.2 - PyTorch with MPS (Apple Silicon) support
- ✅ **datasets** ^3.0.0 - Hugging Face datasets library
- ✅ **diffusers** ^0.31.0 - Image generation models
- ✅ **peft** ^0.16.0 - Parameter-efficient fine-tuning (updated from 1.0.0)
- ✅ **sentencepiece** - Tokenizer support (installed via pip)
- ✅ **accelerate** ^1.5.0 - Training acceleration
- ✅ **evaluate** ^0.4.0 - For bias/toxicity metrics
- ❌ **bitsandbytes** - Not available on macOS ARM64 (commented out)

### Resolved Issues

1. **PEFT Version Issue**
   - Problem: `peft ^1.0.0` doesn't exist
   - Solution: Updated to `peft ^0.16.0` (latest available version)

2. **Bitsandbytes on macOS ARM64**
   - Problem: bitsandbytes 0.46.0 not available for Apple Silicon
   - Solution: Commented out in pyproject.toml as optional dependency
   - Note: INT8/INT4 quantization features will be limited on macOS

3. **Python Version**
   - Problem: Python 3.12.10 not yet available in pyenv
   - Solution: Using Python 3.12.9 (latest available 3.12.x)

4. **Poetry Lock File**
   - Problem: Initial lock file had incompatible dependencies
   - Solution: Regenerated with `poetry lock --no-cache`

### Known Limitations

1. **PyTorch Security Warning**
   - Current: PyTorch 2.2.2 installed
   - Issue: Security vulnerability CVE-2025-32434 requires torch 2.6+
   - Impact: Some models (e.g., cardiffnlp/twitter-roberta-base-sentiment-latest) won't load unless they use safetensors format
   - Workaround: Using fallback models that work with current PyTorch version

2. **Bitsandbytes on macOS**
   - INT8/INT4 quantization features not available
   - Flash Attention not supported on Apple Silicon
   - MPS (Metal Performance Shaders) is used instead of CUDA

### Testing the Environment

Run these commands to verify your setup:

```bash
# Test environment - checks all imports
task test-env

# Run quick demo - tests basic pipeline
task quick-start

# List all available tasks
task --list
```

### Device Support
- **macOS (Apple Silicon)**: Uses MPS device for GPU acceleration
- **Linux/Windows with NVIDIA GPU**: Would use CUDA with full quantization support
- **CPU**: Fallback option for all platforms

### Alternative Models

Since some modern models require newer PyTorch versions, here are compatible alternatives:

- **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english` (works)
- **Text Generation**: `gpt2` or `distilgpt2` (compatible)
- **NER**: `dslim/bert-base-NER` (should work)
- **Question Answering**: `distilbert-base-uncased-distilled-squad` (compatible)

### Running the Tutorial Notebook

```bash
# Activate the environment
poetry shell

# Run Jupyter notebook
poetry run jupyter notebook notebooks/tutorial.ipynb
```

## Next Steps

1. Run the tutorial notebook to explore all Chapter 8 examples
2. Check out the demonstration scripts in `src/`
3. Read the improved documentation in `docs/art_08i.md`
4. For production use, consider upgrading PyTorch when compatible versions are available
5. If you need full quantization features, use a Linux system with NVIDIA GPU

The project is ready for development and experimentation!