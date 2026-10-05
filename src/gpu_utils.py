"""
GPU Acceleration & Diagnostic Utilities for TensorFlow / Keras.

Provides central functions to:
  1. Detect NVIDIA GPU physical devices.
  2. Enable dynamic VRAM memory growth to prevent TensorFlow from allocating 100% VRAM upfront.
  3. Report clear diagnostic messages regarding GPU availability without fake reporting.
"""
from __future__ import annotations

import sys
from typing import Any

import tensorflow as tf


def setup_gpu(verbose: bool = True) -> tuple[bool, list[str]]:
    """
    Detect available physical GPUs and configure memory growth.

    Returns:
        (gpu_available: bool, gpu_names: list[str])
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        if verbose:
            print(f"[GPU Setup] Error querying physical devices: {e}")
        return False, []

    if not gpus:
        if verbose:
            print("[GPU Setup] ℹ No GPU detected by TensorFlow. Running on CPU.")
        return False, []

    gpu_names = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            if verbose:
                print(f"[GPU Setup] Note on {gpu.name}: {e}")
        except Exception as e:
            if verbose:
                print(f"[GPU Setup] Could not set memory growth for {gpu.name}: {e}")

        gpu_names.append(gpu.name)

    if verbose:
        details = get_gpu_details()
        print("\n" + "=" * 60)
        print("  NVIDIA GPU ACCELERATION DETECTED")
        print("=" * 60)
        print(f"  TensorFlow Version : {tf.__version__}")
        print(f"  CUDA Built Status  : {tf.test.is_built_with_cuda()}")
        print(f"  GPU Count          : {len(gpus)}")
        for idx, dev_name in enumerate(details.get("device_names", gpu_names)):
            print(f"  GPU [{idx}]              : {dev_name}")
        print("  Memory Growth      : Enabled (dynamic allocation)")
        print("=" * 60 + "\n")

    return True, gpu_names


def get_gpu_details() -> dict[str, Any]:
    """Retrieve detailed GPU and framework diagnostic information."""
    gpus = tf.config.list_physical_devices("GPU")
    device_names = []
    
    # Attempt to get detailed device names via device_lib if possible
    try:
        from tensorflow.python.client import device_lib
        local_devices = device_lib.list_local_devices()
        for d in local_devices:
            if d.device_type == "GPU":
                # Extract clean name from physical_device_desc if present
                desc = d.physical_device_desc
                if "name:" in desc:
                    name_part = desc.split("name:")[1].split(",")[0].strip()
                    device_names.append(f"{d.name} ({name_part})")
                else:
                    device_names.append(d.name)
    except Exception:
        device_names = [g.name for g in gpus]

    if not device_names and gpus:
        device_names = [g.name for g in gpus]

    return {
        "tf_version": tf.__version__,
        "cuda_built": tf.test.is_built_with_cuda(),
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "physical_devices": [g.name for g in gpus],
        "device_names": device_names,
    }


if __name__ == "__main__":
    setup_gpu(verbose=True)
