# Unified Results JSON Prompt

Use the following prompt with another model when you want it to convert that
model's training/testing outputs into the same JSON format as your other runs.

```text
You are helping me normalize experiment outputs from different cardiac MRI segmentation models into one unified JSON format.

Return exactly one valid JSON object and nothing else.

Requirements:
1. Do not invent numbers. If a value is missing, use null.
2. Keep all metric values as raw numbers, not strings.
3. Use the exact key names below.
4. Paths should be strings.
5. If a section is not available, still include it with null values or empty arrays/objects as appropriate.

JSON schema:
{
  "model_name": "short model name, e.g. unet / swin_unet / nnunet",
  "run_name": "experiment name",
  "checkpoint": "path to best checkpoint or null",
  "dataset": {
    "name": "dataset name, e.g. ACDC",
    "training_root": "path or null",
    "testing_root": "path or null",
    "num_training_patients": 0,
    "num_testing_patients": 0
  },
  "task": {
    "type": "medical_segmentation",
    "classes": [
      "background",
      "rv_cavity",
      "myocardium",
      "lv_cavity"
    ],
    "target_curve_class": "rv_cavity"
  },
  "training": {
    "best_epoch": null,
    "epochs_trained": null,
    "best_val_macro_dice_fg": null,
    "best_val_loss": null
  },
  "validation": {
    "macro_dice_fg": null,
    "dice_rv_cavity": null,
    "dice_myocardium": null,
    "dice_lv_cavity": null
  },
  "testing_labeled_frames": {
    "macro_dice_fg": null,
    "dice_rv_cavity": null,
    "dice_myocardium": null,
    "dice_lv_cavity": null,
    "num_patients_evaluated": null,
    "num_labeled_frame_volumes": null,
    "summary_json": null
  },
  "testing_curve": {
    "num_patients_processed": null,
    "mean_abs_max_minus_ed": null,
    "mean_abs_min_minus_es": null,
    "summary_json": null
  },
  "artifacts": {
    "history_json": null,
    "per_patient_metrics_csv": null,
    "per_labeled_frame_metrics_csv": null,
    "curve_summary_json": null
  },
  "notes": "short free-text note or null"
}

If you are given partial inputs:
- Fill what you can from the provided logs/files.
- Use null for unavailable fields.
- Never rename keys.
```
