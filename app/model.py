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
        self.device = Config.DEVICE
        self._load_model()
    
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
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **Config.get_model_kwargs()
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
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
            gen_kwargs = Config.get_generation_kwargs()
            gen_kwargs.update(generation_kwargs)
            
            # Ensure we have max_new_tokens instead of max_length for better control
            if "max_length" in gen_kwargs and "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_length")
            
            # Format messages for Qwen using chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for older tokenizers
                text = self._format_messages_manual(messages)
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            input_length = model_inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
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
            logger.error(f"Error during generation: {str(e)}")
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

