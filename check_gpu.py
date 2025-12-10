"""Script to check GPU and CUDA availability."""
import sys

print("=" * 60)
print("GPU and CUDA Availability Check")
print("=" * 60)

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test CUDA
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        result = test_tensor * 2
        print(f"\n✓ CUDA test passed! Tensor computation works.")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n✗ CUDA test failed: {e}")
        print("  CUDA is detected but not working properly.")
else:
    print("\n✗ CUDA is not available")
    print("\nPossible reasons:")
    print("  1. PyTorch was installed without CUDA support")
    print("     Solution: Install PyTorch with CUDA:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  2. NVIDIA drivers are not installed")
    print("     Solution: Install NVIDIA drivers for your GPU")
    print("\n  3. GPU is not accessible")
    print("     Solution: Check GPU with: nvidia-smi")

# Check nvidia-smi
print("\n" + "=" * 60)
print("Checking nvidia-smi...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi works:")
        print(result.stdout[:500])  # First 500 chars
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found (NVIDIA drivers may not be installed)")
except subprocess.TimeoutExpired:
    print("✗ nvidia-smi timed out")
except Exception as e:
    print(f"✗ Error running nvidia-smi: {e}")

print("\n" + "=" * 60)

