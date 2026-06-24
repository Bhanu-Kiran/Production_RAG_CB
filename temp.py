# pyrefly: ignore [missing-import]
import torch
from sentence_transformers import CrossEncoder

print("=" * 50)
print("🛡️ RAG HARDWARE ACCELERATION CHECK")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"🔹 CUDA GPU Detection : {'✅ SUCCESS (GPU Enabled)' if cuda_available else '❌ FAILED (Running on CPU)'}")

if cuda_available:
    print(f"🔹 Active GPU Device  : {torch.cuda.get_device_name(0)}")
    
    # Quick model loading test
    print("\n🔹 Testing model allocation to GPU...")
    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        print("✅ Success! Local Cross-Encoder loaded directly into GPU VRAM.")
    except Exception as e:
        print(f"❌ Model allocation failed: {e}")
else:
    print("\n⚠️ Warning: Running local models on your CPU will cause latency lags.")

print("=" * 50)