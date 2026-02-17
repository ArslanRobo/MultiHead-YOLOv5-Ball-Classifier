# PyTorch Model Inference Instructions

## Files Needed

1. `best_classifier.pt` - Trained classifier model
2. `baseline_best.pt` - Baseline detection model  
3. `yolo_with_classifier.py` - Model architecture
4. `pytorch_inference_example.py` - Inference script

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python pyyaml
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

### 2. Run Inference

```bash
python pytorch_inference_example.py --image path/to/your/image.jpg
```

### 3. Custom Device (GPU)

```bash
python pytorch_inference_example.py --image path/to/image.jpg --device cuda
```

## Output Example

```
================================================================================
RUNNING INFERENCE
================================================================================

Image: basketball.jpg

🎯 Prediction: Basketball
   Confidence: 87.45%

📊 All Probabilities:
   Basketball     : 87.45%
   Football       :  8.32%
   Tennis Ball    :  4.23%

📈 Logits: [1.2345, -0.4567, -0.8901]

================================================================================
✅ INFERENCE COMPLETE
================================================================================
```

## Classes

- 0: Basketball
- 1: Football
- 2: Tennis Ball

## Notes

- Model expects 640x640 input (auto-resized)
- Images are automatically normalized to [0, 1]
- Architecture file must be in the same directory or provide path with `--arch`
