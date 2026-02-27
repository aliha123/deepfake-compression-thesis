import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
