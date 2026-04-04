from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from inference_utils import CLASS_NAMES, infer_4d_volume, load_model_from_checkpoint, load_nii_array, parse_info_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D U-Net inference on a 4D cine MRI volume.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=1, help="Class used to estimate ED/ES by area; default 1=RV cavity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.num_classes, device)

    nii, volume_4d = load_nii_array(args.input)
    predictions = infer_4d_volume(model, volume_4d, args.image_size, device)

    target_class_areas = [
        int((predictions[:, :, :, frame_idx] == args.target_class).sum())
        for frame_idx in range(predictions.shape[3])
    ]

    pred_nii = nib.Nifti1Image(predictions, affine=nii.affine, header=nii.header)
    nib.save(pred_nii, str(args.output_dir / "pred_masks_4d.nii.gz"))

    ed_frame = int(np.argmax(target_class_areas)) + 1
    es_frame = int(np.argmin(target_class_areas)) + 1
    info_cfg = parse_info_cfg(args.input.parent / "Info.cfg")
    summary = {
        "input_shape": list(volume_4d.shape),
        "target_class": args.target_class,
        "target_class_name": CLASS_NAMES.get(args.target_class, f"class_{args.target_class}"),
        "target_class_areas": target_class_areas,
        "estimated_ed_frame_1_based": ed_frame,
        "estimated_es_frame_1_based": es_frame,
    }
    if "ED" in info_cfg:
        summary["reference_ed_frame_1_based"] = info_cfg["ED"]
    if "ES" in info_cfg:
        summary["reference_es_frame_1_based"] = info_cfg["ES"]
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
