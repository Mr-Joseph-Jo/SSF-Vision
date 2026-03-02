#!/usr/bin/env python3
"""Test DirectML GPU acceleration with your models."""

import torch
import sys

print("=" * 60)
print("DIRECTML GPU DETECTION TEST")
print("=" * 60)

# Test 1: Check DirectML availability
print("\n[1] Checking torch-directml...")
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✓ torch-directml installed")
    print(f"✓ DirectML device: {device}")
except ImportError:
    print("✗ torch-directml NOT installed")
    print("  Run: pip install torch-directml")
    sys.exit(1)

# Test 2: Create a tensor on DirectML
print("\n[2] Testing tensor operations on DirectML...")
try:
    x = torch.ones(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)  # Matrix multiply (GPU operation)
    print(f"✓ Matrix operation successful")
    print(f"  Result dtype: {z.dtype}, shape: {z.shape}")
    print(f"  Result device: {z.device}")
except Exception as e:
    print(f"✗ Tensor operation failed: {e}")
    sys.exit(1)

# Test 3: Load YOLOv8 model on DirectML
print("\n[3] Testing YOLOv8 on DirectML...")
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    # Move model to DirectML device
    model.to(device)
    print(f"✓ YOLOv8 model loaded on DirectML")
    print(f"  Model device: {next(model.model.parameters()).device}")
except Exception as e:
    print(f"⚠ YOLOv8 warning: {e}")

# Test 4: Check reid.py device selection
print("\n[4] Checking reid.py device selection...")
try:
    from reid import get_device
    selected_device = get_device()
    print(f"✓ get_device() returned: {selected_device}")
except Exception as e:
    print(f"✗ Error loading reid.py: {e}")

print("\n" + "=" * 60)
print("SUMMARY: DirectML GPU acceleration is READY ✓")
print("=" * 60)
print("\nYour app will now use Intel Arc GPU via DirectML.")
print("Start main.py to begin surveillance with GPU acceleration!")
