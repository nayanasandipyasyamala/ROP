"""Run an end-to-end ROP screening and stage classification on a retinal fundus image.

Pipeline:
  1. Screening (Task A): Evaluates if the eye is Normal vs ROP.
  2. Staging   (Task B): If ROP is detected, classifies the severity and stage:
     - Mild (Stage 1 / 2)
     - Moderate (Stage 3)
     - Severe (Stage 4 / 5 / A)

Expected project layout:
D:\\UROP\\
  predict.py
  rop_severity_model.pt
  checkpoints\\taskA_resnet50_ce_fold0_best.pt
  src\\models.py
  src\\preprocessing.py
  configs\\config.py

Usage:
  python predict.py "path\\to\\image.jpg"
  python predict.py   # Prompts interactively for image path
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
        "Could not import project modules. Make sure this script is placed in D:\\UROP "
        f"and src/ is accessible.\nImport error: {exc}"
    ) from exc


# Checkpoints
DEFAULT_TASK_A_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "taskA_resnet50_ce_fold0_best.pt"
DEFAULT_TASK_B_CHECKPOINT = PROJECT_ROOT / "rop_severity_model.pt"

# Clinical stage taxonomy
STAGE_NAMES = [
    "Mild (Stage 1 / 2)",
    "Moderate (Stage 3)",
    "Severe (Stage 4 / 5 / A)",
]
IMAGE_SIZE = 224


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model state_dict from either full trainer checkpoint or bare state_dict."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid state_dict in checkpoint: {checkpoint_path}")

    # Strip DataParallel prefix if present
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def predict(
    image_path: Path,
    task_a_checkpoint: Path = DEFAULT_TASK_A_CHECKPOINT,
    task_b_checkpoint: Path = DEFAULT_TASK_B_CHECKPOINT,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(img_size=IMAGE_SIZE, is_training=False)

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"The file is not a readable image: {image_path}") from exc

    input_tensor = transform(image).unsqueeze(0).to(device)

    print("\n========== ROP DETECTION & STAGE ANALYSIS ==========")
    print(f"Image : {image_path}")
    print(f"Device: {device}\n")

    # ---------------------------------------------------------
    # Stage 1: Binary Screening (Normal vs ROP)
    # ---------------------------------------------------------
    has_task_a = task_a_checkpoint.is_file()
    is_rop = True
    screening_text = "Screening model not found; evaluating severity directly."

    if has_task_a:
        model_a = get_model(
            model_name="resnet50",
            num_classes=2,
            pretrained=False,
            dropout=0.2,
        )
        load_checkpoint_weights(model_a, task_a_checkpoint, device)
        model_a.to(device)
        model_a.eval()

        with torch.inference_mode():
            logits_a = model_a(input_tensor)
            probs_a = F.softmax(logits_a, dim=1)[0].detach().cpu()

        prob_normal = probs_a[0].item() * 100
        prob_rop = probs_a[1].item() * 100

        if prob_rop > prob_normal:
            is_rop = True
            print(f"Screening Result: ROP Detected ({prob_rop:.2f}%)")
        else:
            is_rop = False
            print(f"Screening Result: Normal (No ROP) ({prob_normal:.2f}%)")
            print("Stage Analysis  : No active ROP staging required.")
            print("====================================================")
            print("Research-use output only. Not for standalone clinical diagnosis.")
            return
    else:
        print(f"Notice: {screening_text}")

    # ---------------------------------------------------------
    # Stage 2: Staging & Severity Classification (if ROP detected)
    # ---------------------------------------------------------
    if not task_b_checkpoint.is_file():
        raise FileNotFoundError(
            f"Severity model checkpoint not found at: {task_b_checkpoint}\n"
            "Please ensure rop_severity_model.pt is placed in the project root."
        )

    model_b = get_model(
        model_name="resnet50",
        num_classes=3,
        pretrained=False,
        dropout=0.2,
    )
    load_checkpoint_weights(model_b, task_b_checkpoint, device)
    model_b.to(device)
    model_b.eval()

    with torch.inference_mode():
        logits_b = model_b(input_tensor)
        probs_b = F.softmax(logits_b, dim=1)[0].detach().cpu()

    predicted_stage_idx = int(torch.argmax(probs_b).item())
    predicted_stage_name = STAGE_NAMES[predicted_stage_idx]
    predicted_stage_prob = probs_b[predicted_stage_idx].item() * 100

    print(f"Predicted Stage : {predicted_stage_name} ({predicted_stage_prob:.2f}%)")
    print("\nStage Probabilities:")
    for stage_name, prob in zip(STAGE_NAMES, probs_b.tolist()):
        print(f"  {stage_name:24s}: {prob * 100:6.2f}%")
    print("====================================================")
    print("Research-use output only. Not for standalone clinical diagnosis.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end ROP screening and stage classification on a retinal image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a retinal fundus image (JPG, JPEG, or PNG).",
    )
    parser.add_argument(
        "--taskA-checkpoint",
        default=str(DEFAULT_TASK_A_CHECKPOINT),
        help=f"Path to the Task A binary checkpoint (default: {DEFAULT_TASK_A_CHECKPOINT}).",
    )
    parser.add_argument(
        "--taskB-checkpoint",
        default=str(DEFAULT_TASK_B_CHECKPOINT),
        help=f"Path to the Task B severity checkpoint (default: {DEFAULT_TASK_B_CHECKPOINT}).",
    )
    args = parser.parse_args()

    image_value = args.image or input("Enter retinal image path: ").strip().strip('"')
    if not image_value:
        print("No image path was provided.", file=sys.stderr)
        return 2

    try:
        predict(
            Path(image_value).expanduser(),
            task_a_checkpoint=Path(args.taskA_checkpoint).expanduser(),
            task_b_checkpoint=Path(args.taskB_checkpoint).expanduser(),
        )
    except Exception as exc:
        print(f"\nPrediction failed: {exc}", file=sys.stderr)
        return 1
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
