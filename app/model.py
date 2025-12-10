"""Model loading and inference for Qwen chatbot."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict
import logging
from app.config import Config

logger = logging.getLogger(__name__)


class QwenChatbot:
    """Qwen chatbot model wrapper."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # Ensure device is valid
        self.device = self._get_valid_device(Config.DEVICE)
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _get_valid_device(self, device: str) -> str:
        """Get a valid device string, ensuring CUDA is available if requested."""
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if device in ("cuda", "auto"):
            if cuda_available:
                logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available. Using CPU instead.")
                logger.warning("This could mean:")
                logger.warning("  1. PyTorch was installed without CUDA support")
                logger.warning("  2. NVIDIA drivers are not installed")
                logger.warning("  3. GPU is not accessible")
                return "cpu"
        elif device == "cpu":
            return "cpu"
        else:
            logger.warning(f"Unknown device '{device}', using CPU")
            return "cpu"
    
    def _load_model(self):
        """Load the Qwen model and tokenizer."""
        try:
            logger.info(f"Loading model from {Config.MODEL_PATH}...")
            
            # Check if local model exists
            import os
            if os.path.exists(Config.MODEL_LOCAL_DIR) and os.path.isdir(Config.MODEL_LOCAL_DIR):
                model_path = Config.MODEL_LOCAL_DIR
                logger.info(f"Using local model at {model_path}")
            else:
                model_path = Config.MODEL_PATH
                logger.info(f"Downloading model from Hugging Face: {model_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                **Config.get_model_kwargs()
            )
            
            # Load model
            logger.info("Loading model (this may take a while)...")
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                **Config.get_model_kwargs()
            }
            
            # Set dtype and device_map - force CPU if CUDA not actually available
            cuda_available = torch.cuda.is_available()
            if self.device == "cuda" and cuda_available:
                try:
                    # Test CUDA actually works
                    test_tensor = torch.tensor([1.0]).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    load_kwargs["dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
                    logger.info("Using CUDA for model loading")
                except Exception as e:
                    logger.warning(f"CUDA test failed: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    cuda_available = False
                    load_kwargs["dtype"] = torch.float32
                    load_kwargs["device_map"] = None
            else:
                load_kwargs["dtype"] = torch.float32
                load_kwargs["device_map"] = None
                self.device = "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Force move to CPU if CUDA not available
            if not cuda_available or self.device == "cpu":
                logger.info("Moving model to CPU...")
                self.model = self.model.to("cpu")
                self.device = "cpu"
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **generation_kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            # Prepare generation parameters
            gen_kwargs = Config.get_generation_kwargs().copy()
            gen_kwargs.update(generation_kwargs)
            
            # Convert max_length to max_new_tokens if present
            if "max_length" in gen_kwargs:
                if "max_new_tokens" not in gen_kwargs:
                    # max_length includes input, so we need to calculate max_new_tokens
                    # For now, use a reasonable default or the provided value
                    gen_kwargs["max_new_tokens"] = min(gen_kwargs.pop("max_length"), 2048)
                else:
                    gen_kwargs.pop("max_length")  # Remove max_length if max_new_tokens exists
            
            # Ensure max_new_tokens is set
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 512  # Default reasonable value
            
            # Remove max_length from kwargs to avoid conflicts
            gen_kwargs.pop("max_length", None)
            
            # Format messages for Qwen using chat template
            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback for older tokenizers
                    text = self._format_messages_manual(messages)
            except Exception as e:
                logger.warning(f"Error applying chat template: {e}, using manual formatting")
                text = self._format_messages_manual(messages)
            
            # Tokenize - ensure we use the correct device
            # Double-check CUDA availability at runtime
            device = "cpu"  # Default to CPU
            if self.device == "cuda":
                try:
                    if torch.cuda.is_available():
                        # Test if CUDA actually works
                        test = torch.tensor([1.0]).cuda()
                        del test
                        device = "cuda"
                    else:
                        logger.warning("CUDA not available at runtime, using CPU")
                        self.device = "cpu"
                except Exception as e:
                    logger.warning(f"CUDA error at runtime: {e}. Using CPU.")
                    self.device = "cpu"
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
            input_length = model_inputs.input_ids.shape[1]
            
            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Ensure model is on the correct device
            if device == "cpu":
                self.model = self.model.to("cpu")
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs
                )
            
            # Extract only the new tokens (response)
            generated_ids = generated_ids[0][input_length:]
            
            # Decode response
            response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            raise
    
    def _format_messages_manual(self, messages: List[Dict[str, str]]) -> str:
        """Manual message formatting fallback."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        return formatted
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None


# Global model instance
_chatbot_instance: Optional[QwenChatbot] = None


def get_chatbot() -> QwenChatbot:
    """Get or create the global chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = QwenChatbot()
    return _chatbot_instance

