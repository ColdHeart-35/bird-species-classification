"""
GPU Diagnostic Utility — Bird Species Classification.

Checks and reports hardware GPU detection, CUDA availability, and TensorFlow setup.

Usage:
    python -m src.gpu_check
"""
from __future__ import annotations

import sys
import tensorflow as tf

from src.gpu_utils import get_gpu_details, setup_gpu


def main() -> None:
    print("=" * 60)
    print("  Bird Vision ML Project — GPU Diagnostic Check")
    print("=" * 60)
    print(f"Python Executable   : {sys.executable}")
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"TensorFlow Version  : {tf.__version__}")
    print(f"Built with CUDA     : {tf.test.is_built_with_cuda()}")
    
    gpu_available, gpus = setup_gpu(verbose=False)
    details = get_gpu_details()

    print("-" * 60)
    if gpu_available:
        print("STATUS              : ✓ GPU Acceleration AVAILABLE")
        print(f"GPUs Detected       : {details['gpu_count']}")
        for idx, name in enumerate(details['device_names']):
            print(f"  Device [{idx}]        : {name}")
        print("Memory Growth       : Configured for dynamic VRAM allocation")
        print("\nModel training and inference will automatically use the NVIDIA GPU.")
    else:
        print("STATUS              : ℹ No GPU Detected by TensorFlow")
        print("Execution Mode      : CPU Fallback")
        print("\nTroubleshooting tips for NVIDIA GPU on Fedora/Linux:")
        print("  1. Verify NVIDIA drivers are active: run `nvidia-smi` in terminal.")
        print("  2. Ensure CUDA/cuDNN PyPI dependencies are installed in your virtualenv:")
        print('     pip install "tensorflow[and-cuda]"')
        print("  3. Check system library paths if using system CUDA toolkit.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
