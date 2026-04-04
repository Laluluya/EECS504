from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from model import UNet2D


CLASS_NAMES = {
    0: "background",
    1: "rv_cavity",
    2: "myocardium",
    3: "lv_cavity",
}

LABEL_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [255, 191, 0],
        [65, 105, 225],
    ],
    dtype=np.uint8,
)


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        return image - mean
    return (image - mean) / std


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, len(LABEL_COLORS) - 1)
    return LABEL_COLORS[mask]


def load_model_from_checkpoint(
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
) -> UNet2D:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet2D(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def infer_4d_volume(
    model: UNet2D,
    volume_4d: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    if volume_4d.ndim != 4:
        raise ValueError(f"Expected 4D input, got shape {volume_4d.shape}")

    original_h, original_w, depth, num_frames = volume_4d.shape
    predictions = np.zeros((original_h, original_w, depth, num_frames), dtype=np.uint8)

    with torch.no_grad():
        for frame_idx in range(num_frames):
            for slice_idx in range(depth):
                image = normalize_slice(volume_4d[:, :, slice_idx, frame_idx])
                tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
                tensor = F.interpolate(
                    tensor,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                logits = model(tensor)
                pred = torch.argmax(logits, dim=1, keepdim=True).float()
                pred = F.interpolate(pred, size=(original_h, original_w), mode="nearest")
                predictions[:, :, slice_idx, frame_idx] = pred.squeeze().cpu().numpy().astype(np.uint8)

    return predictions


def load_nii_array(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    nii = nib.load(str(path))
    return nii, nii.get_fdata().astype(np.float32)


def parse_info_cfg(info_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    if not info_path.exists():
        return result
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in {"ED", "ES"}:
            result[key] = int(value)
    return result
