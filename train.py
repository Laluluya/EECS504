from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

PROJECT_CACHE_DIR = Path(tempfile.gettempdir()) / "medical_segmentation_cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for subdir in ["mpl", "wandb", "wandb_cache", "wandb_config"]:
    (PROJECT_CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "mpl"))
os.environ.setdefault("WANDB_DIR", str(PROJECT_CACHE_DIR / "wandb"))
os.environ.setdefault("WANDB_CACHE_DIR", str(PROJECT_CACHE_DIR / "wandb_cache"))
os.environ.setdefault("WANDB_CONFIG_DIR", str(PROJECT_CACHE_DIR / "wandb_config"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CardiacSliceDataset, discover_examples, patient_ids_from_examples, split_examples_by_patient
from inference_utils import CLASS_NAMES, colorize_mask
from model import UNet2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D U-Net on the local cardiac MRI dataset.")
    parser.add_argument("--data-roots", nargs="+", default=["training"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-patients", nargs="*", default=None)
    parser.add_argument("--max-train-slices", type=int, default=None)
    parser.add_argument("--max-val-slices", type=int, default=None)
    parser.add_argument("--log-pred-every", type=int, default=1)
    parser.add_argument("--num-preview-samples", type=int, default=3)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="medical-segmentation")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--save-dir", type=Path, default=Path("runs/baseline"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multiclass_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def compute_macro_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    preds = torch.argmax(logits, dim=1)
    class_scores: list[float] = []
    for cls in range(1, num_classes):
        pred_mask = (preds == cls).float()
        true_mask = (targets == cls).float()
        intersection = torch.sum(pred_mask * true_mask)
        denom = torch.sum(pred_mask) + torch.sum(true_mask)
        if denom.item() == 0:
            class_scores.append(1.0)
        else:
            class_scores.append(float((2.0 * intersection + eps) / (denom + eps)))
    return float(sum(class_scores) / len(class_scores))


def compute_class_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    metrics: dict[str, float] = {}
    for cls in range(1, num_classes):
        pred_mask = (preds == cls).float()
        true_mask = (targets == cls).float()
        intersection = torch.sum(pred_mask * true_mask)
        denom = torch.sum(pred_mask) + torch.sum(true_mask)
        if denom.item() == 0:
            score = 1.0
        else:
            score = float((2.0 * intersection + eps) / (denom + eps))
        metrics[f"dice_{CLASS_NAMES.get(cls, cls)}"] = score
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            ce = F.cross_entropy(logits, masks)
            dice = multiclass_dice_loss(logits, masks, num_classes=num_classes)
            loss = ce + dice

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_dice += compute_macro_dice(logits.detach(), masks, num_classes=num_classes)
        total_batches += 1

    return total_loss / max(total_batches, 1), total_dice / max(total_batches, 1)


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, history: list[dict[str, float | int]]) -> dict:
    serialized_args = {}
    for key, value in vars(args).items():
        serialized_args[key] = str(value) if isinstance(value, Path) else value
    return {
        "model_state": model.state_dict(),
        "args": serialized_args,
        "history": history,
    }


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Recreate the conda env with `conda env create -f environment.yml` "
            "or install it into your active conda env."
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        settings=wandb.Settings(
            start_method="thread",
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_viewer=True,
        ),
    )
    return run


def render_preview_figure(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(colorize_mask(mask))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(colorize_mask(pred))
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_and_log_previews(
    model: nn.Module,
    dataset: CardiacSliceDataset,
    device: torch.device,
    epoch: int,
    save_dir: Path,
    num_classes: int,
    preview_count: int,
    wandb_run,
) -> None:
    preview_dir = save_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    images_for_wandb = []
    preview_total = min(preview_count, len(dataset))
    for sample_idx in range(preview_total):
        sample = dataset[sample_idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        mask_tensor = sample["mask"]
        with torch.no_grad():
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

        image_np = sample["image"].squeeze(0).cpu().numpy()
        mask_np = mask_tensor.cpu().numpy()
        pred_np = pred.numpy()
        patient_id = str(sample["patient_id"])
        slice_idx = int(sample["slice_idx"])
        frame_id = str(sample["frame_id"])
        title = f"epoch={epoch} patient={patient_id} frame={frame_id} slice={slice_idx}"
        output_path = preview_dir / f"epoch_{epoch:03d}_{patient_id}_{frame_id}_slice{slice_idx:02d}.png"
        render_preview_figure(image_np, mask_np, pred_np, title, output_path)

        if wandb_run is not None:
            import wandb

            images_for_wandb.append(wandb.Image(str(output_path), caption=title))

    if wandb_run is not None and images_for_wandb:
        wandb_run.log({"val/previews": images_for_wandb, "epoch": epoch})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = discover_examples([Path(root) for root in args.data_roots])
    train_examples, val_examples = split_examples_by_patient(
        examples,
        val_ratio=args.val_ratio,
        explicit_val_patients=args.val_patients,
    )

    train_dataset = CardiacSliceDataset(
        train_examples,
        image_size=args.image_size,
        augment=True,
        max_samples=args.max_train_slices,
    )
    val_dataset = CardiacSliceDataset(
        val_examples,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_val_slices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet2D(num_classes=args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    wandb_run = maybe_init_wandb(args)

    summary = {
        "device": str(device),
        "num_examples": len(examples),
        "num_patients": len(patient_ids_from_examples(examples)),
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "train_patients": sorted({example.patient_id for example in train_examples}),
        "val_patients": sorted({example.patient_id for example in val_examples}),
    }
    print(json.dumps(summary, indent=2))

    best_val_dice = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, device, args.num_classes)
        val_loss, val_dice = run_epoch(model, val_loader, None, device, args.num_classes)

        first_val_batch = next(iter(val_loader))
        with torch.no_grad():
            first_val_logits = model(first_val_batch["image"].to(device))
        class_dice = compute_class_dice(
            first_val_logits.detach().cpu(),
            first_val_batch["mask"],
            args.num_classes,
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_dice_fg": train_dice,
            "val_loss": val_loss,
            "val_macro_dice_fg": val_dice,
        }
        epoch_stats.update(class_dice)
        history.append(epoch_stats)
        print(json.dumps(epoch_stats))

        if wandb_run is not None:
            wandb_run.log(epoch_stats)

        latest_path = args.save_dir / "latest.pt"
        torch.save(checkpoint_payload(model, args, history), latest_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = args.save_dir / "best.pt"
            torch.save(checkpoint_payload(model, args, history), best_path)

        if args.log_pred_every > 0 and epoch % args.log_pred_every == 0:
            save_and_log_previews(
                model=model,
                dataset=val_dataset,
                device=device,
                epoch=epoch,
                save_dir=args.save_dir,
                num_classes=args.num_classes,
                preview_count=args.num_preview_samples,
                wandb_run=wandb_run,
            )

    with (args.save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
