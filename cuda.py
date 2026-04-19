import os
# Force CUDA visibility FIRST
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quick test
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print("GPU tensor multiplication: SUCCESS")
else:
    print("\n❌ CUDA still not available!")
    print("\nPossible reasons:")
    print("1. CPU-only PyTorch installed (reinstall with --index-url)")
    print("2. NVIDIA drivers outdated or corrupted")
    print("3. Laptop using Intel GPU instead of NVIDIA")
    print("4. Conflicting CUDA installations")