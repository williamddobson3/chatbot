"""Test script to diagnose chatbot issues."""
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ Transformers installed")
    except ImportError as e:
        print(f"✗ Transformers not installed: {e}")
        return False
    
    try:
        from app.config import Config
        print("✓ Config loaded")
        print(f"  - Model path: {Config.MODEL_PATH}")
        print(f"  - Device: {Config.DEVICE}")
        print(f"  - Local dir: {Config.MODEL_LOCAL_DIR}")
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if model can be loaded."""
    print("\nTesting model loading...")
    try:
        from app.model import get_chatbot
        print("Loading model (this may take a while)...")
        chatbot = get_chatbot()
        if chatbot.is_loaded():
            print("✓ Model loaded successfully!")
            return chatbot
        else:
            print("✗ Model failed to load")
            return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chat(chatbot):
    """Test chat functionality."""
    print("\nTesting chat...")
    try:
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        print("Sending test message...")
        response = chatbot.chat(messages, max_new_tokens=50)
        print(f"✓ Chat test successful!")
        print(f"Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"✗ Chat test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Chatbot Diagnostic Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed. Please install dependencies.")
        sys.exit(1)
    
    # Test model loading
    chatbot = test_model_loading()
    if chatbot is None:
        print("\n✗ Model loading failed. Check logs and configuration.")
        sys.exit(1)
    
    # Test chat
    if not test_chat(chatbot):
        print("\n✗ Chat test failed. Check model and generation parameters.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    main()

