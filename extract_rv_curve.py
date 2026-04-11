from __future__ import annotations

import argparse
import csv
import json
import os
import re
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


def _find_labeled_frame_gt_path(patient_dir: Path, patient_id: str, frame_1_based: int) -> Path | None:
    """Resolve e.g. patient101_frame14_gt.nii.gz for frame index from Info.cfg (1-based)."""
    rx = re.compile(rf"^{re.escape(patient_id)}_frame(\d+)_gt\.nii\.gz$")
    for path in sorted(patient_dir.glob(f"{patient_id}_frame*_gt.nii.gz")):
        m = rx.match(path.name)
        if m and int(m.group(1)) == frame_1_based:
            return path
    return None


def _manual_gt_target_area_pixels(gt_path: Path, target_class: int) -> int:
    data = nib.load(str(gt_path)).get_fdata()
    mask = np.asarray(np.rint(data), dtype=np.int64)
    return int((mask == target_class).sum())


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
    max_frame = int(np.argmax(frame_areas)) + 1
    min_frame = int(np.argmin(frame_areas)) + 1
    info_cfg = parse_info_cfg(input_path.parent / "Info.cfg")

    csv_path = output_dir / "frame_areas.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_1_based", "area_pixels"])
        for frame_idx, area in zip(frame_indices, frame_areas):
            writer.writerow([frame_idx, area])

    patient_dir = input_path.parent
    patient_id = patient_dir.name

    curve_path = output_dir / "area_curve.png"
    plt.figure(figsize=(9, 4))
    plt.plot(frame_indices, frame_areas, marker="o", linewidth=1.8)
    plt.scatter(
        [max_frame],
        [frame_areas[max_frame - 1]],
        color="red",
        marker="^",
        s=120,
        label=f"pred max frame={max_frame}",
    )
    plt.scatter(
        [min_frame],
        [frame_areas[min_frame - 1]],
        color="blue",
        marker="v",
        s=120,
        label=f"pred min frame={min_frame}",
    )
    if "ED" in info_cfg and 1 <= info_cfg["ED"] <= len(frame_areas):
        gt_ed_frame = info_cfg["ED"]
        ed_gt_path = _find_labeled_frame_gt_path(patient_dir, patient_id, gt_ed_frame)
        if ed_gt_path is not None:
            gt_ed_area = _manual_gt_target_area_pixels(ed_gt_path, target_class)
            ed_label = f"manual GT ED frame={gt_ed_frame}, area={gt_ed_area}"
        else:
            gt_ed_area = frame_areas[gt_ed_frame - 1]
            ed_label = f"pred @ ED frame={gt_ed_frame} (no labeled *_gt.nii.gz), area={gt_ed_area}"
        plt.scatter(
            [gt_ed_frame],
            [gt_ed_area],
            facecolors="none",
            edgecolors="darkred",
            marker="s",
            s=130,
            linewidths=2.2,
            label=ed_label,
        )
    if "ES" in info_cfg and 1 <= info_cfg["ES"] <= len(frame_areas):
        gt_es_frame = info_cfg["ES"]
        es_gt_path = _find_labeled_frame_gt_path(patient_dir, patient_id, gt_es_frame)
        if es_gt_path is not None:
            gt_es_area = _manual_gt_target_area_pixels(es_gt_path, target_class)
            es_label = f"manual GT ES frame={gt_es_frame}, area={gt_es_area}"
        else:
            gt_es_area = frame_areas[gt_es_frame - 1]
            es_label = f"pred @ ES frame={gt_es_frame} (no labeled *_gt.nii.gz), area={gt_es_area}"
        plt.scatter(
            [gt_es_frame],
            [gt_es_area],
            facecolors="none",
            edgecolors="navy",
            marker="D",
            s=130,
            linewidths=2.2,
            label=es_label,
        )
    plt.xlabel("Frame (1-based)")
    plt.ylabel("RV cavity area (pixels)")
    plt.title(f"Area curve for {CLASS_NAMES.get(target_class, target_class)}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

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
    if "ED" in info_cfg and 1 <= info_cfg["ED"] <= len(frame_areas):
        ed_f = info_cfg["ED"]
        summary["reference_ed_frame_1_based"] = ed_f
        ed_gt_path = _find_labeled_frame_gt_path(patient_dir, patient_id, ed_f)
        if ed_gt_path is not None:
            summary["reference_ed_area_pixels"] = _manual_gt_target_area_pixels(ed_gt_path, target_class)
            summary["reference_ed_area_pixels_pred"] = frame_areas[ed_f - 1]
        else:
            summary["reference_ed_area_pixels"] = frame_areas[ed_f - 1]
    if "ES" in info_cfg and 1 <= info_cfg["ES"] <= len(frame_areas):
        es_f = info_cfg["ES"]
        summary["reference_es_frame_1_based"] = es_f
        es_gt_path = _find_labeled_frame_gt_path(patient_dir, patient_id, es_f)
        if es_gt_path is not None:
            summary["reference_es_area_pixels"] = _manual_gt_target_area_pixels(es_gt_path, target_class)
            summary["reference_es_area_pixels_pred"] = frame_areas[es_f - 1]
        else:
            summary["reference_es_area_pixels"] = frame_areas[es_f - 1]
    if "ED" in info_cfg and "ES" in info_cfg and all(1 <= info_cfg[key] <= len(frame_areas) for key in ("ED", "ES")):
        summary["frame_error_vs_reference"] = {
            "max_minus_ed": max_frame - info_cfg["ED"],
            "min_minus_es": min_frame - info_cfg["ES"],
            "abs_max_minus_ed": abs(max_frame - info_cfg["ED"]),
            "abs_min_minus_es": abs(min_frame - info_cfg["ES"]),
        }

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
