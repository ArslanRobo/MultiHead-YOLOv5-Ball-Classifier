#!/usr/bin/env python3
# ==============================================================================
# RKNN INT8 Quantization Script
# ==============================================================================
# Prerequisites:
# - Install RKNN-Toolkit2: pip install rknn-toolkit2
# - Download from: https://github.com/airockchip/rknn-toolkit2
# ==============================================================================

from rknn.api import RKNN
import numpy as np
from pathlib import Path

print("=" * 80)
print("🤖 RKNN INT8 QUANTIZATION")
print("=" * 80)

# Configuration
ONNX_MODEL = 'ball_classifier.onnx'
RKNN_MODEL = 'ball_classifier_int8.rknn'
CALIBRATION_LIST = 'calibration_list.txt'

# Target platform (change to your target device)
TARGET_PLATFORM = 'rk3588'  # Options: rk3588, rk3568, rk3566, rk3562, rv1126, etc.

print(f"\n⚙️  Configuration:")
print(f"   ONNX Model: {ONNX_MODEL}")
print(f"   RKNN Model: {RKNN_MODEL}")
print(f"   Target Platform: {TARGET_PLATFORM}")
print(f"   Quantization: INT8")

# Create RKNN object
rknn = RKNN(verbose=True)

# Configure RKNN
print("\n🔧 Configuring RKNN...")
ret = rknn.config(
    mean_values=[[0, 0, 0]],              # Already normalized in preprocessing
    std_values=[[255, 255, 255]],         # Scale back from [0,1] to [0,255]
    target_platform=TARGET_PLATFORM,
    quantized_dtype='asymmetric_quantized-8',  # INT8 quantization
    quantized_algorithm='normal',          # Options: 'normal', 'mmse' (better accuracy)
    quantized_method='channel',            # Per-channel quantization
    optimization_level=3                   # Optimization level (0-3)
)

if ret != 0:
    print('❌ Config failed!')
    exit(ret)
print("✅ Configuration successful!")

# Load ONNX model
print(f"\n📥 Loading ONNX model...")
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('❌ Load ONNX model failed!')
    exit(ret)
print("✅ ONNX model loaded!")

# Build RKNN model with INT8 quantization
print(f"\n🔄 Building RKNN model with INT8 quantization...")
print(f"   Using calibration data from: {CALIBRATION_LIST}")
print(f"   This may take a few minutes...")

ret = rknn.build(
    do_quantization=True,
    dataset=CALIBRATION_LIST,
    rknn_batch_size=1
)

if ret != 0:
    print('❌ Build RKNN model failed!')
    exit(ret)
print("✅ RKNN model built successfully!")

# Export RKNN model
print(f"\n💾 Exporting RKNN model...")
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('❌ Export RKNN model failed!')
    exit(ret)

model_size = Path(RKNN_MODEL).stat().st_size / 1024 / 1024
print(f"✅ RKNN model exported!")
print(f"   Saved to: {RKNN_MODEL}")
print(f"   File size: {model_size:.2f} MB")

# Initialize runtime (will fail if not on RK board)
print(f"\n🚀 Initializing RKNN runtime...")
ret = rknn.init_runtime()
if ret != 0:
    print('⚠️  Init runtime failed (normal if not running on RK board)')
    print('   Model is ready for deployment on target hardware')
else:
    print("✅ Runtime initialized!")

    # Test inference
    print("\n🧪 Testing quantized model...")
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = rknn.inference(inputs=[dummy_input])

    print(f"   Detection output shape: {outputs[0].shape}")
    print(f"   Classification output shape: {outputs[1].shape}")
    print(f"   Classification output: {outputs[1]}")
    print(f"   Predicted class: {np.argmax(outputs[1])}")

    class_names = {0: 'Basketball', 1: 'Football', 2: 'Tennis Ball'}
    print(f"   Predicted ball type: {class_names[np.argmax(outputs[1])]}")

# Cleanup
rknn.release()

print("\n" + "=" * 80)
print("✅ RKNN CONVERSION COMPLETE!")
print("=" * 80)
print(f"\n📦 Deliverables:")
print(f"   • ONNX Model: {ONNX_MODEL}")
print(f"   • RKNN Model: {RKNN_MODEL}")
print(f"   • Model size: {model_size:.2f} MB")
print(f"   • Ready for deployment on {TARGET_PLATFORM}!")
print(f"\n🎯 Next Steps:")
print(f"   1. Copy {RKNN_MODEL} to your RK board")
print(f"   2. Use RKNN-Toolkit-Lite2 for inference on device")
print(f"   3. Integrate with your application")
