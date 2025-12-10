"""Configuration management for the chatbot application."""
import os
from typing import Optional
from dotenv import load_dotenv
import torch

load_dotenv()


def _get_device() -> str:
    """Auto-detect device, fallback to CPU if CUDA not available."""
    device_env = os.getenv("DEVICE", "auto")
    
    if device_env.lower() == "auto":
        # Auto-detect: use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_env.lower() == "cuda":
        # Check if CUDA is actually available
        if torch.cuda.is_available():
            return "cuda"
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    else:
        return device_env.lower()


class Config:
    """Application configuration."""
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-72B-Instruct")
    MODEL_LOCAL_DIR: str = os.getenv("MODEL_LOCAL_DIR", "./models/Qwen2.5-72B-Instruct")
    
    # Device Configuration (auto-detects CUDA availability)
    DEVICE: str = _get_device()
    
    # Generation Parameters
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "50"))
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Hugging Face Token
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    
    @classmethod
    def get_model_kwargs(cls) -> dict:
        """Get model loading kwargs."""
        kwargs = {}
        if cls.HF_TOKEN:
            kwargs["token"] = cls.HF_TOKEN
        return kwargs
    
    @classmethod
    def get_generation_kwargs(cls) -> dict:
        """Get text generation kwargs."""
        return {
            "max_new_tokens": min(cls.MAX_LENGTH, 2048),  # Use max_new_tokens instead of max_length
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "top_k": cls.TOP_K,
            "do_sample": True,
            # Note: pad_token_id and eos_token_id are set explicitly in model.py
        }

