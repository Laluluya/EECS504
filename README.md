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
python train.py --epochs 20 --batch-size 8 --save-dir runs/full
```

## Predict a 4D cine volume

```bash
python predict_4d.py \
  --checkpoint runs/full/best.pt \
  --input "testing/patient101/patient101_4d.nii.gz" \
  --output-dir runs/patient101_pred
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

## Weights & Biases

Online logging:

```bash
python train.py \
  --epochs 20 \
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
  --epochs 20 \
  --batch-size 8 \
  --save-dir runs/wandb_offline \
  --use-wandb \
  --wandb-mode offline \
  --log-pred-every 5
```

When preview logging is enabled, the script also saves local segmentation previews under `runs/.../previews/`.
