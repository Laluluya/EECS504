from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import CardiacSliceDataset, discover_examples
from inference_utils import CLASS_NAMES, load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dice only on frames that have labels, e.g. ED/ES "
            "patientXXX_frameYY.nii.gz + patientXXX_frameYY_gt.nii.gz pairs."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("testing"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/testing_labeled_eval"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def empty_stats(num_classes: int) -> dict[str, list[float]]:
    return {
        "intersection": [0.0 for _ in range(num_classes)],
        "denom": [0.0 for _ in range(num_classes)],
    }


def update_stats(stats: dict[str, list[float]], pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> None:
    for cls in range(1, num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        stats["intersection"][cls] += float((pred_mask & target_mask).sum().item())
        stats["denom"][cls] += float(pred_mask.sum().item() + target_mask.sum().item())


def dice_from_stats(stats: dict[str, list[float]], num_classes: int, eps: float = 1e-6) -> dict[str, float]:
    result: dict[str, float] = {}
    foreground_scores: list[float] = []
    for cls in range(1, num_classes):
        denom = stats["denom"][cls]
        if denom == 0:
            dice = 1.0
        else:
            dice = (2.0 * stats["intersection"][cls] + eps) / (denom + eps)
        result[f"dice_{CLASS_NAMES.get(cls, f'class_{cls}')}"] = dice
        foreground_scores.append(dice)
    result["macro_dice_fg"] = sum(foreground_scores) / len(foreground_scores)
    return result


def write_overall_plot(overall_metrics: dict[str, float], output_path: Path) -> None:
    keys = [
        "dice_rv_cavity",
        "dice_myocardium",
        "dice_lv_cavity",
        "macro_dice_fg",
    ]
    labels = ["RV", "Myo", "LV", "Macro FG"]
    values = [overall_metrics[key] for key in keys]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=["#dc143c", "#ffbf00", "#4169e1", "#2f4f4f"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Dice")
    plt.title("Overall Dice on Labeled Frames")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_per_patient_plot(per_patient_rows: list[dict[str, float | str]], output_path: Path) -> None:
    patient_ids = [str(row["patient_id"]) for row in per_patient_rows]
    macro_scores = [float(row["macro_dice_fg"]) for row in per_patient_rows]

    plt.figure(figsize=(max(10, len(patient_ids) * 0.22), 4.5))
    plt.plot(range(len(patient_ids)), macro_scores, marker="o", linewidth=1.6, color="#2f4f4f")
    plt.xticks(range(len(patient_ids)), patient_ids, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Macro Dice (FG)")
    plt.title("Per-Patient Macro Dice on Labeled Frames")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.num_classes, device)

    examples = discover_examples([args.data_root])
    dataset = CardiacSliceDataset(examples, image_size=args.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    overall_stats = empty_stats(args.num_classes)
    patient_stats = defaultdict(lambda: empty_stats(args.num_classes))
    frame_stats = defaultdict(lambda: empty_stats(args.num_classes))
    labeled_frames = {(example.patient_id, example.frame_id) for example in examples}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"]
            preds = torch.argmax(model(images), dim=1).cpu()

            for idx in range(preds.shape[0]):
                patient_id = batch["patient_id"][idx]
                frame_id = batch["frame_id"][idx]
                pred = preds[idx]
                target = targets[idx]

                update_stats(overall_stats, pred, target, args.num_classes)
                update_stats(patient_stats[patient_id], pred, target, args.num_classes)
                update_stats(frame_stats[(patient_id, frame_id)], pred, target, args.num_classes)

    per_patient_path = args.output_dir / "per_patient_metrics.csv"
    per_patient_rows: list[dict[str, float | str]] = []
    with per_patient_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["patient_id", "macro_dice_fg"] + [
            f"dice_{CLASS_NAMES.get(cls, f'class_{cls}')}" for cls in range(1, args.num_classes)
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id in sorted(patient_stats):
            row = {"patient_id": patient_id}
            row.update(dice_from_stats(patient_stats[patient_id], args.num_classes))
            writer.writerow(row)
            per_patient_rows.append(row)

    per_frame_path = args.output_dir / "per_labeled_frame_metrics.csv"
    with per_frame_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["patient_id", "frame_id", "macro_dice_fg"] + [
            f"dice_{CLASS_NAMES.get(cls, f'class_{cls}')}" for cls in range(1, args.num_classes)
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id, frame_id in sorted(frame_stats):
            row = {"patient_id": patient_id, "frame_id": frame_id}
            row.update(dice_from_stats(frame_stats[(patient_id, frame_id)], args.num_classes))
            writer.writerow(row)

    patient_macro_scores = [
        dice_from_stats(patient_stats[patient_id], args.num_classes)["macro_dice_fg"]
        for patient_id in patient_stats
    ]
    overall_metrics = dice_from_stats(overall_stats, args.num_classes)

    overall_json_path = args.output_dir / "overall_metrics.json"
    per_patient_json_path = args.output_dir / "per_patient_metrics.json"
    overall_plot_path = args.output_dir / "overall_dice.png"
    per_patient_plot_path = args.output_dir / "per_patient_macro_dice.png"

    overall_json = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": str(device),
        "num_slices": len(examples),
        "num_patients": len(patient_stats),
        "num_labeled_frame_volumes": len(labeled_frames),
        "metrics": overall_metrics,
        "mean_macro_dice_fg_over_patients": sum(patient_macro_scores) / len(patient_macro_scores),
    }
    with overall_json_path.open("w", encoding="utf-8") as f:
        json.dump(overall_json, f, indent=2)

    with per_patient_json_path.open("w", encoding="utf-8") as f:
        json.dump(per_patient_rows, f, indent=2)

    write_overall_plot(overall_metrics, overall_plot_path)
    write_per_patient_plot(per_patient_rows, per_patient_plot_path)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": str(device),
        "num_slices": len(examples),
        "num_patients": len(patient_stats),
        "num_labeled_frame_volumes": len(labeled_frames),
        "overall": overall_metrics,
        "mean_macro_dice_fg_over_patients": sum(patient_macro_scores) / len(patient_macro_scores),
        "artifacts": {
            "overall_metrics_json": str(overall_json_path),
            "per_patient_metrics_json": str(per_patient_json_path),
            "per_patient_metrics": str(per_patient_path),
            "per_labeled_frame_metrics": str(per_frame_path),
            "overall_dice_plot": str(overall_plot_path),
            "per_patient_macro_dice_plot": str(per_patient_plot_path),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
