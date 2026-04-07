# Cardiac MRI U-Net Baseline

Minimal 2D U-Net training code for the official ACDC-style cardiac MRI dataset from the Human Heart Project:

[Human Heart Project dataset page](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)

## Dataset expectation

Download the official dataset and organize it with the standard split:

- `training/`
- `testing/`

The training script uses `training/` by default.

Each patient directory should contain:

- `patientXXX_frameYY.nii.gz`
- `patientXXX_frameYY_gt.nii.gz`
- `Info.cfg`

The `training/` set is used for supervised learning. Only ED/ES frames with `*_gt.nii.gz` are used for training.

The `testing/` set is used for inference on full cine volumes such as:

- `patientXXX_4d.nii.gz`
- `Info.cfg`

## Install

```bash
conda env create -f environment.yml
conda activate eecs504-medical
```

## Smoke test

```bash
python train.py --epochs 1 --batch-size 4 --max-train-slices 24 --max-val-slices 12 --save-dir runs/smoke
```

## Full-ish local run

```bash
python train.py --epochs 50 --batch-size 8 --save-dir runs/full
```

## Predict a 4D cine volume

```bash
python predict_4d.py \
  --checkpoint runs/full/best.pt \
  --input "testing/patient101/patient101_4d.nii.gz" \
  --output-dir runs/patient101_pred
```

## Evaluate Labeled Testing Frames

ACDC labels only the ED/ES frames. This evaluates Dice on the available
`patientXXX_frameYY.nii.gz` + `patientXXX_frameYY_gt.nii.gz` pairs and does not
require labels for every frame in `patientXXX_4d.nii.gz`.

```bash
python evaluate_labeled_frames.py \
  --checkpoint runs/full/best.pt \
  --data-root testing \
  --output-dir runs/testing_labeled_eval
```

## RV area curve

```bash
python extract_rv_curve.py \
  --checkpoint runs/full/best.pt \
  --input "testing/patient101/patient101_4d.nii.gz" \
  --output-dir runs/patient101_rv_curve
```

This writes:

- `pred_masks_4d.nii.gz`
- `frame_areas.csv`
- `area_curve.png`
- `summary.json`

Batch extraction for every testing patient:

```bash
python batch_extract_rv_curves.py \
  --checkpoint runs/full/best.pt \
  --data-root testing \
  --output-dir runs/testing_rv_curves
```

## Weights & Biases

Online logging:

```bash
python train.py \
  --epochs 50 \
  --batch-size 8 \
  --save-dir runs/wandb_run \
  --use-wandb \
  --wandb-project medical-segmentation \
  --wandb-name acdc-unet-rvmyolv \
  --log-pred-every 5
```

Offline logging:

```bash
python train.py \
  --epochs 50 \
  --batch-size 8 \
  --save-dir runs/wandb_offline \
  --use-wandb \
  --wandb-mode offline \
  --log-pred-every 5
```

When preview logging is enabled, the script also saves local segmentation previews under `runs/.../previews/`.
