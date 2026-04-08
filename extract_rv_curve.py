from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

PROJECT_CACHE_DIR = Path(tempfile.gettempdir()) / "medical_segmentation_cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_CACHE_DIR / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "mpl"))

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from inference_utils import CLASS_NAMES, infer_4d_volume, load_model_from_checkpoint, load_nii_array, parse_info_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict 4D masks and extract an RV area curve.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=1, help="Default 1 = RV cavity.")
    return parser.parse_args()


def write_rv_curve_outputs(
    predictions: np.ndarray,
    input_path: Path,
    output_dir: Path,
    target_class: int,
    input_shape: tuple[int, ...],
    include_pred_path: bool = False,
) -> dict:
    frame_indices = list(range(1, predictions.shape[3] + 1))
    frame_areas = [
        int((predictions[:, :, :, frame_idx - 1] == target_class).sum())
        for frame_idx in frame_indices
    ]
    ed_frame = int(np.argmax(frame_areas)) + 1
    es_frame = int(np.argmin(frame_areas)) + 1

    csv_path = output_dir / "frame_areas.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_1_based", "area_pixels"])
        for frame_idx, area in zip(frame_indices, frame_areas):
            writer.writerow([frame_idx, area])

    curve_path = output_dir / "area_curve.png"
    plt.figure(figsize=(9, 4))
    plt.plot(frame_indices, frame_areas, marker="o", linewidth=1.8)
    plt.scatter([ed_frame], [frame_areas[ed_frame - 1]], color="red", label=f"max frame={ed_frame}")
    plt.scatter([es_frame], [frame_areas[es_frame - 1]], color="blue", label=f"min frame={es_frame}")
    plt.xlabel("Frame (1-based)")
    plt.ylabel("Predicted area (pixels)")
    plt.title(f"Area curve for {CLASS_NAMES.get(target_class, target_class)}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

    info_cfg = parse_info_cfg(args.input.parent / "Info.cfg")
    summary = {
        "patient_id": input_path.parent.name,
        "input": str(input_path),
        "input_shape": list(input_shape),
        "target_class": target_class,
        "target_class_name": CLASS_NAMES.get(target_class, f"class_{target_class}"),
        "frame_areas": frame_areas,
        "estimated_max_frame_1_based": max_frame,
        "estimated_min_frame_1_based": min_frame,
        "artifacts": {
            "frame_areas_csv": str(csv_path),
            "area_curve_png": str(curve_path),
        },
    }
    if include_pred_path:
        summary["artifacts"]["pred_masks_4d"] = str(output_dir / "pred_masks_4d.nii.gz")
    if "ED" in info_cfg:
        summary["reference_ed_frame_1_based"] = info_cfg["ED"]
    if "ES" in info_cfg:
        summary["reference_es_frame_1_based"] = info_cfg["ES"]

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.num_classes, device)
    nii, volume_4d = load_nii_array(args.input)
    predictions = infer_4d_volume(model, volume_4d, args.image_size, device)

    nib.save(
        nib.Nifti1Image(predictions, affine=nii.affine, header=nii.header),
        str(args.output_dir / "pred_masks_4d.nii.gz"),
    )
    summary = write_rv_curve_outputs(
        predictions=predictions,
        input_path=args.input,
        output_dir=args.output_dir,
        target_class=args.target_class,
        input_shape=volume_4d.shape,
        include_pred_path=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
