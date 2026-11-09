"""Run a trained 3-class ROP severity ResNet50 checkpoint on one fundus image.

Expected project layout:
D:\\UROP\\
  predict.py
  rop_severity_model.pt
  src\\models.py
  src\\preprocessing.py
  configs\\config.py

Usage from the project root:
  python predict.py "test_images\\sample.jpg"

If no image path is supplied, the script prompts for one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models import get_model
    from src.preprocessing import get_transforms
except Exception as exc:
    raise SystemExit(
        "Could not import the existing project modules. Make sure this script is "
        "saved in the D:\\UROP project root and that src\\models.py, "
        "src\\preprocessing.py, and configs\\config.py are present.\n"
        f"Import error: {exc}"
    ) from exc


CLASS_NAMES = ["Mild", "Moderate", "Severe"]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "rop_severity_model.pt"
IMAGE_SIZE = 224


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Place rop_severity_model.pt in the D:\\UROP project root, "
            "or provide --checkpoint with its full path."
        )

    # Trainer checkpoints store weights under 'model_state_dict'.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Also allow a plain state_dict if the checkpoint was saved directly.
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("The checkpoint does not contain a valid model state_dict.")

    # Handle checkpoints saved from a DataParallel-wrapped model.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def predict(image_path: Path, checkpoint_path: Path) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Match the Kaggle training architecture: ResNet50, 3 outputs, dropout 0.2.
    # pretrained=False avoids downloading ImageNet weights; the trained checkpoint
    # supplies all learned model parameters.
    model = get_model(
        model_name="resnet50",
        num_classes=3,
        pretrained=False,
        dropout=0.2,
    )
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    # Use the project's deterministic validation/test transforms (no augmentation).
    transform = get_transforms(img_size=IMAGE_SIZE, is_training=False)

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"The file is not a readable image: {image_path}") from exc

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_index]

    print("\n========== ROP SEVERITY PREDICTION ==========")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print("Model output scores:")
    for class_name, score in zip(CLASS_NAMES, probabilities.tolist()):
        print(f"  {class_name:9s}: {score * 100:6.2f}%")
    print("=============================================")
    print(
        "Research-use output only. These scores are model probabilities, "
        "not calibrated clinical certainty or a medical diagnosis."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict Mild, Moderate, or Severe ROP using the trained ResNet50 model."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a retinal fundus image (JPG, JPEG, or PNG).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to the trained checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    args = parser.parse_args()

    image_value = args.image or input("Enter the retinal image path: ").strip().strip('"')
    if not image_value:
        print("No image path was provided.", file=sys.stderr)
        return 2

    try:
        predict(Path(image_value).expanduser(), Path(args.checkpoint).expanduser())
    except Exception as exc:
        print(f"\nPrediction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
