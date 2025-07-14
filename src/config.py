"""Configuration management for workflow examples."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Device configuration
def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Environment settings
DEVICE = os.getenv("DEFAULT_DEVICE", get_device())
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Helper function for pipeline device configuration
def get_pipeline_device():
    """Get device ID for HuggingFace pipelines."""
    if DEVICE == "cuda":
        return 0  # GPU 0
    elif DEVICE == "mps":
        return 0  # MPS device 0
    else:
        return -1  # CPU

# Model settings
DEFAULT_SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
DEFAULT_NER_MODEL = os.getenv(
    "NER_MODEL",
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)
DEFAULT_CLASSIFICATION_MODEL = os.getenv(
    "CLASSIFICATION_MODEL",
    "facebook/bart-large-mnli"
)
DEFAULT_GENERATION_MODEL = os.getenv(
    "GENERATION_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# Data settings
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Optimization settings
ENABLE_QUANTIZATION = os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))
ENABLE_FLASH_ATTENTION = os.getenv("ENABLE_FLASH_ATTENTION", "true").lower() == "true"

# Synthetic data settings
SYNTHETIC_DATA_PATH = Path(os.getenv("SYNTHETIC_DATA_PATH", "./synthetic"))
VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "0.85"))

# Create directories
DATA_PATH.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
SYNTHETIC_DATA_PATH.mkdir(exist_ok=True)

# Hugging Face settings
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


class Config:
    """Centralized configuration class for easy access."""

    # Device configuration
    DEVICE = torch.device(DEVICE)

    # Batch and model settings
    BATCH_SIZE = BATCH_SIZE
    MAX_LENGTH = MAX_LENGTH

    # Model names
    DEFAULT_SENTIMENT_MODEL = DEFAULT_SENTIMENT_MODEL
    DEFAULT_NER_MODEL = DEFAULT_NER_MODEL
    DEFAULT_CLASSIFICATION_MODEL = DEFAULT_CLASSIFICATION_MODEL
    DEFAULT_GENERATION_MODEL = DEFAULT_GENERATION_MODEL

    # Paths
    DATA_PATH = DATA_PATH
    CACHE_DIR = CACHE_DIR
    SYNTHETIC_DATA_PATH = SYNTHETIC_DATA_PATH

    # Other settings
    NUM_WORKERS = NUM_WORKERS
    ENABLE_QUANTIZATION = ENABLE_QUANTIZATION
    QUANTIZATION_BITS = QUANTIZATION_BITS
    ENABLE_FLASH_ATTENTION = ENABLE_FLASH_ATTENTION
    VALIDATION_THRESHOLD = VALIDATION_THRESHOLD
