from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import torch

from extract_rv_curve import write_rv_curve_outputs
from inference_utils import infer_4d_volume, load_model_from_checkpoint, load_nii_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LV area-curve extraction for every testing patient.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("testing"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/testing_lv_curves"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=3, help="Default 3 = LV cavity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.num_classes, device)

    patient_dirs = sorted(path for path in args.data_root.glob("patient*") if path.is_dir())
    summaries = []
    skipped = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        volume_path = patient_dir / f"{patient_id}_4d.nii.gz"
        if not volume_path.exists():
            skipped.append({"patient_id": patient_id, "reason": f"missing {volume_path.name}"})
            continue

        nii, volume_4d = load_nii_array(volume_path)
        predictions = infer_4d_volume(model, volume_4d, args.image_size, device)
        patient_output_dir = args.output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        nib.save(
            nib.Nifti1Image(predictions, affine=nii.affine, header=nii.header),
            str(patient_output_dir / "pred_masks_4d.nii.gz"),
        )
        summary = write_rv_curve_outputs(
            predictions=predictions,
            input_path=volume_path,
            output_dir=patient_output_dir,
            target_class=args.target_class,
            input_shape=volume_4d.shape,
            include_pred_path=True,
        )
        summaries.append(summary)

    aggregate = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": str(device),
        "num_patient_dirs": len(patient_dirs),
        "num_processed": len(summaries),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "patients": summaries,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
