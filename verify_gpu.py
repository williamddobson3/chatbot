"""Quick script to verify GPU is working and model can generate."""
import sys
import torch
from app.model import get_chatbot
from app.config import Config
import time

print("=" * 60)
print("GPU and Model Verification")
print("=" * 60)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Current Device: {Config.DEVICE}")

# Load model
print("\n2. Loading model...")
try:
    chatbot = get_chatbot()
    if chatbot.is_loaded():
        print("   ✓ Model loaded successfully")
        
        # Check where model is
        try:
            first_param = next(chatbot.model.parameters())
            if first_param.is_cuda:
                print(f"   ✓ Model is on GPU")
            else:
                print(f"   ⚠ Model is on CPU")
        except:
            print("   ⚠ Could not determine model device")
    else:
        print("   ✗ Model failed to load")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

# Test generation
print("\n3. Testing generation...")
try:
    messages = [{"role": "user", "content": "Say hello in one word."}]
    print("   Starting generation (this may take a while)...")
    start_time = time.time()
    
    response = chatbot.chat(messages, max_new_tokens=50)
    
    elapsed = time.time() - start_time
    print(f"   ✓ Generation successful!")
    print(f"   Response: {response[:100]}...")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Tokens/sec: {len(response.split()) / elapsed:.2f}")
    
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All checks passed!")
print("=" * 60)

