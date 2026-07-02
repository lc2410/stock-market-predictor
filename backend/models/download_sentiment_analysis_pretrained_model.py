import os
from huggingface_hub import snapshot_download

def download_finbert():
    print("Downloading ProsusAI/finbert model weights locally...")
    # Save the model inside backend/models/finbert_weights
    model_dir = os.path.join(os.path.dirname(__file__), "finbert_weights")
    os.makedirs(model_dir, exist_ok=True)
    
    # Download the PyTorch safetensors and configs, skip heavy unused formats
    snapshot_download(repo_id="ProsusAI/finbert", local_dir=model_dir, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
    print(f"Model downloaded successfully to: {model_dir}")
    print("Your application will now load the model locally without pinging the HuggingFace API!")

if __name__ == "__main__":
    download_finbert()
