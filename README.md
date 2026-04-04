# Cardiac MRI U-Net Baseline

Minimal 2D U-Net training code for the local ACDC-style cardiac MRI dataset in this workspace.

## Dataset expectation

The scripts scan both:

- `Resources/`
- `Resources (1)/`

Each patient directory should contain:

- `patientXXX_frameYY.nii.gz`
- `patientXXX_frameYY_gt.nii.gz`
- `Info.cfg`

Only ED/ES frames with `*_gt.nii.gz` are used for supervised training.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
  --input "Resources/patient101/patient101_4d.nii.gz" \
  --output-dir runs/patient101_pred
```

## RV area curve

```bash
python extract_rv_curve.py \
  --checkpoint runs/full/best.pt \
  --input "Resources/patient101/patient101_4d.nii.gz" \
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
