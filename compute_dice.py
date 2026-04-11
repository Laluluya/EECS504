from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from inference_utils import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-class Dice between two segmentation masks.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to predicted mask NIfTI.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground-truth mask NIfTI.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Also report background Dice. Default reports foreground classes only.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the Dice summary JSON.",
    )
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    data = nii.get_fdata()
    return np.rint(data).astype(np.int64)


def dice_score(pred: np.ndarray, gt: np.ndarray, cls: int, eps: float = 1e-6) -> float:
    pred_mask = pred == cls
    gt_mask = gt == cls
    denom = float(pred_mask.sum() + gt_mask.sum())
    if denom == 0:
        return 1.0
    intersection = float(np.logical_and(pred_mask, gt_mask).sum())
    return (2.0 * intersection + eps) / (denom + eps)


def main() -> None:
    args = parse_args()
    pred = load_mask(args.pred)
    gt = load_mask(args.gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    start_cls = 0 if args.include_background else 1
    metrics: dict[str, float] = {}
    foreground_scores: list[float] = []

    for cls in range(start_cls, args.num_classes):
        score = dice_score(pred, gt, cls)
        metrics[f"dice_{CLASS_NAMES.get(cls, f'class_{cls}')}"] = score
        if cls != 0:
            foreground_scores.append(score)

    summary = {
        "pred": str(args.pred),
        "gt": str(args.gt),
        "shape": list(pred.shape),
        "num_classes": args.num_classes,
        "metrics": metrics,
    }
    if foreground_scores:
        summary["macro_dice_fg"] = sum(foreground_scores) / len(foreground_scores)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
