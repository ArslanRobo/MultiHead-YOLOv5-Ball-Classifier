
# ==============================================================================
# models/yolo_with_classifier.py
# ==============================================================================
# Multi-Head YOLO: Frozen Detection Model + Trainable Classification Head
# THIS IS THE EXACT ARCHITECTURE FROM PART 2 TRAINING NOTEBOOK
# ==============================================================================

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Multi-scale classification head."""
    
    def __init__(self, nc_cls=3, input_channels=[64, 128, 256]):
        super().__init__()
        self.nc_cls = nc_cls
        
        # Create a classifier for each scale
        self.classifiers = nn.ModuleList()
        
        for channels in input_channels:
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, nc_cls)
            )
            self.classifiers.append(classifier)
    
    def forward(self, features):
        """Forward pass using multi-scale features."""
        outputs = []
        for i, feat in enumerate(features):
            out = self.classifiers[i](feat)
            outputs.append(out)
        
        # Average outputs from different scales
        final_output = torch.stack(outputs, dim=0).mean(dim=0)
        return final_output


class ModelWithClassifier(nn.Module):
    """YOLOv5 model with added classification head."""
    
    def __init__(self, detection_model, nc_cls=3, freeze_detection=True):
        """
        Args:
            detection_model: Pretrained YOLOv5 detection model
            nc_cls: Number of classification classes
            freeze_detection: Whether to freeze detection layers
        """
        super().__init__()
        
        self.detection_model = detection_model
        self.nc_cls = nc_cls
        self.features = []  # Store intermediate features
        
        # Freeze detection model (backbone + neck + detection head)
        if freeze_detection:
            for param in self.detection_model.parameters():
                param.requires_grad = False
            print("✅ Froze all detection model parameters")
        
        # Find the indices of P3, P4, P5 layers (outputs from neck going to detection head)
        # In YOLOv5, the Detect layer takes inputs from specific layers
        detect_layer = self.detection_model.model[-1]
        self.feature_indices = detect_layer.f  # Indices of layers that feed into Detect
        
        # Register forward hooks to capture features from these layers
        self.hooks = []
        for idx in self.feature_indices:
            layer = self.detection_model.model[idx]
            hook = layer.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
        
        # Get input channels for classification head
        input_channels = []
        for idx in self.feature_indices:
            # Get output channels from each layer
            layer = self.detection_model.model[idx]
            if hasattr(layer, 'cv3'):  # C3 layer
                ch = layer.cv3.conv.out_channels
            elif hasattr(layer, 'conv'):  # Conv layer
                ch = layer.conv.out_channels
            else:
                ch = 256  # Default fallback
            input_channels.append(ch)
        
        # Add classification head
        self.classifier = ClassificationHead(nc_cls=nc_cls, input_channels=input_channels)
        
        print(f"✅ Added classification head (nc_cls={nc_cls})")
        print(f"   Input channels: {input_channels}")
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture intermediate features."""
        self.features.append(output)
    
    def forward(self, x, get_features=False):
        """
        Forward pass through detection model and classification head.
        
        Args:
            x: Input images [batch, 3, H, W]
            get_features: If True, return features for classification
        """
        # Clear previous features
        self.features = []
        
        # Forward through detection model (hooks will capture features)
        det_output = self.detection_model(x)
        
        # Get classification predictions using captured features
        if get_features or self.training:
            cls_logits = self.classifier(self.features)
            return det_output, cls_logits
        else:
            return det_output
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()
