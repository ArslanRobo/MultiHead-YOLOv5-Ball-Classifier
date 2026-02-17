#!/usr/bin/env python3
# ==============================================================================
# PyTorch Inference Example
# ==============================================================================
# This script shows how to load and run inference with best_classifier.pt
# ==============================================================================

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import sys

# Class names
CLASS_NAMES = {0: 'Basketball', 1: 'Football', 2: 'Tennis Ball'}

def load_model(model_path, baseline_path, architecture_path, device='cpu'):
    """
    Load the trained model with proper architecture.
    
    Args:
        model_path: Path to best_classifier.pt
        baseline_path: Path to baseline_best.pt
        architecture_path: Path to yolo_with_classifier.py
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model ready for inference
    """
    print("=" * 80)
    print("LOADING PYTORCH MODEL")
    print("=" * 80)
    
    # 1. Clone YOLOv5 if needed
    if not Path('yolov5').exists():
        print("\n📥 Cloning YOLOv5...")
        import subprocess
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])
        subprocess.run(['pip', 'install', '-q', '-r', 'yolov5/requirements.txt'])
    
    # 2. Add paths
    sys.path.insert(0, str(Path.cwd()))
    sys.path.append('yolov5')
    
    # 3. Import YOLOv5 components
    from models.yolo import DetectionModel
    
    # 4. Load architecture dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "models.yolo_with_classifier", 
        str(architecture_path)
    )
    custom_module = importlib.util.module_from_spec(spec)
    sys.modules["models.yolo_with_classifier"] = custom_module
    sys.modules["yolo_with_classifier"] = custom_module
    spec.loader.exec_module(custom_module)
    
    ModelWithClassifier = custom_module.ModelWithClassifier
    
    # 5. Load baseline model
    print("\n📥 Loading baseline model...")
    import yaml
    baseline_ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
    
    with open('yolov5/models/yolov5n.yaml', 'r') as f:
        model_cfg = yaml.safe_load(f)
        model_cfg['nc'] = 1  # Single class: ball
    
    baseline_model = DetectionModel(model_cfg)
    baseline_model.load_state_dict(baseline_ckpt['model'].state_dict())
    
    # 6. Create multi-head model
    print("📥 Loading classifier model...")
    model = ModelWithClassifier(
        detection_model=baseline_model,
        nc_cls=3,
        freeze_detection=True
    )
    
    # 7. Load weights
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
    else:
        raise KeyError("Invalid checkpoint format")
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def preprocess_image(image_path, target_size=640):
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to image file
        target_size: Target size for model input
    
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension and convert to tensor
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)
    
    return img_tensor


def run_inference(model, image_path, device='cpu'):
    """
    Run inference on a single image.
    
    Args:
        model: Loaded ModelWithClassifier
        image_path: Path to image file
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    img_tensor = preprocess_image(image_path).to(device)
    
    # Inference
    with torch.no_grad():
        det_output, cls_logits = model(img_tensor, get_features=True)
    
    # Get classification result
    pred_class = cls_logits.argmax(dim=1).item()
    probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
    
    result = {
        'predicted_class': pred_class,
        'class_name': CLASS_NAMES[pred_class],
        'confidence': probs[pred_class],
        'probabilities': {CLASS_NAMES[i]: probs[i] for i in range(3)},
        'logits': cls_logits.cpu().numpy()[0]
    }
    
    return result


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch Model Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='best_classifier.pt', help='Path to model checkpoint')
    parser.add_argument('--baseline', type=str, default='baseline_best.pt', help='Path to baseline model')
    parser.add_argument('--arch', type=str, default='yolo_with_classifier.py', help='Path to architecture file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(
        model_path=args.model,
        baseline_path=args.baseline,
        architecture_path=args.arch,
        device=args.device
    )
    
    # Run inference
    print(f"\n{'=' * 80}")
    print("RUNNING INFERENCE")
    print("=" * 80)
    print(f"\nImage: {args.image}")
    
    result = run_inference(model, args.image, device=args.device)
    
    # Print results
    print(f"\n🎯 Prediction: {result['class_name']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"\n📊 All Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"   {class_name:15s}: {prob*100:5.2f}%")
    print(f"\n📈 Logits: {result['logits']}")
    
    print(f"\n{'=' * 80}")
    print("✅ INFERENCE COMPLETE")
    print("=" * 80)
