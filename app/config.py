"""Configuration management for the chatbot application."""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-72B-Instruct")
    MODEL_LOCAL_DIR: str = os.getenv("MODEL_LOCAL_DIR", "./models/Qwen2.5-72B-Instruct")
    
    # Device Configuration
    DEVICE: str = os.getenv("DEVICE", "cuda")
    
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
            "pad_token_id": None,  # Will be set in model if needed
        }

