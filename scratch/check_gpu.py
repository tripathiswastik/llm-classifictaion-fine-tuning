import torch
print("PyTorch Version:", torch.__version__)
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
    print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / (1024**3))

for pkg in ["transformers", "accelerate", "datasets", "torchvision", "xgboost", "lightgbm", "catboost", "sklearn"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"{pkg}: NOT installed")
