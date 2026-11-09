"""Run trained binary ROP screening (Task A: Normal vs ROP) ResNet50 checkpoint on one fundus image.

Expected project layout:
D:\\UROP\\
  predict_taskA.py
  checkpoints\\taskA_resnet50_ce_fold0_best.pt
  src\\models.py
  src\\preprocessing.py

Usage from the project root:
  python predict_taskA.py "test_images\\sample.jpg"

If no image path is supplied, the script prompts for one interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.preprocessing import get_transforms

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "taskA_resnet50_ce_fold0_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Load the binary classification model
    model = get_model(
        model_name="resnet50",
        num_classes=2,
        pretrained=False,
        dropout=0.2,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Use the same deterministic validation preprocessing
    transform = get_transforms(
        img_size=224,
        is_training=False,
    )

    print("Task A binary model loaded successfully!")
    print(f"Device: {device}")

    if len(sys.argv) > 1:
        image_path_str = sys.argv[1].strip().strip('"')
    else:
        image_path_str = input("\nEnter retinal image path: ").strip().strip('"')

    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"Error: image not found at {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        classes = ["Normal", "ROP"]
        predicted_index = probabilities.argmax().item()

        print("\n========== TASK A PREDICTION ==========")
        print(f"Image: {image_path}")
        print(f"Predicted class: {classes[predicted_index]}")
        for i, name in enumerate(classes):
            print(f"  {name:8s}: {probabilities[i].item() * 100:6.2f}%")
        print("========================================")
        print("Research-use output only. Not for standalone clinical diagnosis.")

    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
