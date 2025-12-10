"""Script to download Qwen model from Hugging Face."""
import os
from huggingface_hub import snapshot_download
from app.config import Config

def download_model():
    """Download the Qwen model."""
    print("🚀 Starting model download...")
    print(f"Model: {Config.MODEL_PATH}")
    print(f"Local directory: {Config.MODEL_LOCAL_DIR}")
    print(f"Size: ~145 GB")
    print("\n⚠️  This will take a while depending on your internet connection.")
    print("The download can be resumed if interrupted.\n")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(Config.MODEL_LOCAL_DIR, exist_ok=True)
        
        # Download model
        model_path = snapshot_download(
            repo_id=Config.MODEL_PATH,
            local_dir=Config.MODEL_LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            **Config.get_model_kwargs()
        )
        
        print(f"\n✅ Model downloaded successfully!")
        print(f"Location: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {str(e)}")
        print("\nYou can also download manually using:")
        print(f"  git lfs install")
        print(f"  git clone https://huggingface.co/{Config.MODEL_PATH}")
        raise

if __name__ == "__main__":
    download_model()

