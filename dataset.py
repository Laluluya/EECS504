from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SliceExample:
    patient_id: str
    image_path: Path
    mask_path: Path
    slice_idx: int
    frame_id: str


def discover_examples(data_roots: Iterable[Path]) -> list[SliceExample]:
    examples: list[SliceExample] = []
    for root in data_roots:
        if not root.exists():
            continue
        for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for mask_path in sorted(patient_dir.glob("*_gt.nii.gz")):
                image_name = mask_path.name.replace("_gt.nii.gz", ".nii.gz")
                image_path = patient_dir / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image for mask: {mask_path}")

                image_nii = nib.load(str(image_path))
                mask_nii = nib.load(str(mask_path))
                if image_nii.shape != mask_nii.shape:
                    raise ValueError(
                        f"Shape mismatch for {image_path} and {mask_path}: "
                        f"{image_nii.shape} vs {mask_nii.shape}"
                    )

                if len(image_nii.shape) != 3:
                    raise ValueError(f"Expected 3D frame volume, got {image_nii.shape} for {image_path}")

                frame_id = image_path.name.replace(".nii.gz", "").split("_")[-1]
                for slice_idx in range(image_nii.shape[2]):
                    examples.append(
                        SliceExample(
                            patient_id=patient_dir.name,
                            image_path=image_path,
                            mask_path=mask_path,
                            slice_idx=slice_idx,
                            frame_id=frame_id,
                        )
                    )
    if not examples:
        raise RuntimeError("No image-mask pairs found under the provided data roots.")
    return examples


def patient_ids_from_examples(examples: Iterable[SliceExample]) -> list[str]:
    return sorted({example.patient_id for example in examples})


def split_examples_by_patient(
    examples: list[SliceExample],
    val_ratio: float = 0.2,
    explicit_val_patients: list[str] | None = None,
) -> tuple[list[SliceExample], list[SliceExample]]:
    patients = patient_ids_from_examples(examples)
    if explicit_val_patients:
        val_patients = set(explicit_val_patients)
    else:
        num_val = max(1, round(len(patients) * val_ratio))
        val_patients = set(patients[-num_val:])

    train_examples = [example for example in examples if example.patient_id not in val_patients]
    val_examples = [example for example in examples if example.patient_id in val_patients]
    if not train_examples or not val_examples:
        raise ValueError(
            "Train/val split is empty. Adjust val_ratio or explicit_val_patients."
        )
    return train_examples, val_examples


def _normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-6:
        return image - mean
    return (image - mean) / std


class CardiacSliceDataset(Dataset):
    def __init__(
        self,
        examples: list[SliceExample],
        image_size: int = 256,
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.examples = examples[:max_samples] if max_samples is not None else examples
        self.image_size = image_size
        self.augment = augment
        self._volume_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        example = self.examples[index]
        image_volume = self._load_cached(example.image_path, np.float32)
        mask_volume = self._load_cached(example.mask_path, np.int64)

        image = image_volume[:, :, example.slice_idx]
        mask = mask_volume[:, :, example.slice_idx]

        image = _normalize_slice(image)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

        image_tensor = image_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[1])
            if torch.rand(1).item() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[0])

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patient_id": example.patient_id,
            "slice_idx": example.slice_idx,
            "frame_id": example.frame_id,
        }

    def _load_cached(self, path: Path, dtype: np.dtype) -> np.ndarray:
        if path not in self._volume_cache:
            self._volume_cache[path] = nib.load(str(path)).get_fdata().astype(dtype)
        return self._volume_cache[path]
