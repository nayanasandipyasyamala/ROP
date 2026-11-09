import torch
import torch.nn as nn
from torchvision import models

def load_inference_model(checkpoint_path, device, num_classes=2):
    """Loads the trained ResNet-50 model from checkpoint, matching training architecture."""
    # Initialize base ResNet-50 model without deprecated 'pretrained' argument
    model = models.resnet50(weights=None)
    
    # Match your training classifier structure (e.g., if Sequential or Dropout was used)
    # If your training script used a Sequential block or dropout for fc, replicate it here:
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model