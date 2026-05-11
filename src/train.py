import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DamageMaskDataset
from metrics import binary_confusion_from_logits, binary_scores
from model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net mask prediction model.")
    parser.add_argument("--train-root", default="/Users/oson/Desktop/ada_internship_proj/dataset/train")
    parser.add_argument("--val-root", default="/Users/oson/Desktop/ada_internship_proj/dataset/val")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-features", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke tests.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    batches = 0

    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(image)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits,
                mask,
                pos_weight=pos_weight,
            )
            loss = bce + dice_loss_from_logits(logits, mask)
            tp, fp, fn = binary_confusion_from_logits(logits.detach(), mask.detach())

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        batches += 1

    scores = binary_scores(total_tp, total_fp, total_fn)
    scores["loss"] = total_loss / max(batches, 1)
    return scores


def estimate_pos_weight(dataset: DamageMaskDataset) -> torch.Tensor:
    positives = 0.0
    total = 0.0
    for sample in dataset:
        mask = sample["mask"]
        positives += float(mask.sum().item())
        total += float(mask.numel())
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return torch.tensor([min(negatives / positives, 20.0)], dtype=torch.float32)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    train_names = sorted(p.name for p in (train_root / "pre-event").glob("*.tif"))
    val_names = sorted(p.name for p in (val_root / "pre-event").glob("*.tif"))
    if args.limit is not None:
        train_names = train_names[: args.limit]
        val_names = val_names[: max(1, min(args.limit, len(val_names)))]
    if not train_names or not val_names:
        raise ValueError("Need matched samples in both train and val folders.")

    train_ds = DamageMaskDataset(train_root, train_names, args.image_size, "binary", augment=True)
    val_ds = DamageMaskDataset(val_root, val_names, args.image_size, "binary", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.backends.mps.is_available() or torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.backends.mps.is_available() or torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")

    model = UNet(in_channels=4, out_channels=1, features=args.base_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = estimate_pos_weight(train_ds).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = run_epoch(model, val_loader, None, device, pos_weight)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "task": "binary",
                    "image_size": args.image_size,
                    "base_features": args.base_features,
                    "val_iou": best_iou,
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "train_names": train_names,
                    "val_names": val_names,
                },
                output_dir / "best_model.pt",
            )
            print(f"saved {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
