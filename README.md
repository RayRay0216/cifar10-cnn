# CIFAR-10 CNN (PyTorch)

A minimal, GPU-ready baseline for CIFAR-10 image classification using PyTorch.

## 1) Environment
- Python 3.12 (your venv: `myenv`)
- PyTorch (CUDA 12.8 build)  
  ```bash
  # create/activate venv (if not yet)
  python3 -m venv myenv
  source myenv/bin/activate
  pip install -U pip

  # GPU (CUDA 12.8). For CPU-only, use the cpu index-url instead.
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128