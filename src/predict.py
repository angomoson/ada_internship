import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DamageMaskDataset
from model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict segmentation masks from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--data-root", default="/Users/oson/Desktop/ada_internship_proj/dataset/test")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--task", choices=["binary", "multiclass"], default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    task = args.task or checkpoint.get("task", "binary")
    image_size = int(checkpoint.get("image_size", 512))
    base_features = int(checkpoint.get("base_features", 32))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out_channels = 1 if task == "binary" else 4
    model = UNet(in_channels=4, out_channels=out_channels, features=base_features)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset = DamageMaskDataset(args.data_root, image_size=image_size, task=task, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            logits = model(images)

            for i, name in enumerate(batch["name"]):
                original_size = Image.open(Path(args.data_root) / "target" / name).size[::-1]
                resized = F.interpolate(
                    logits[i : i + 1].cpu(),
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
                if task == "binary":
                    pred = (torch.sigmoid(resized)[0, 0] >= args.threshold).numpy().astype(np.uint8)
                else:
                    pred = torch.argmax(resized, dim=1)[0].numpy().astype(np.uint8)

                Image.fromarray(pred).save(output_dir / name)

    print(f"Wrote predictions to {output_dir}")


if __name__ == "__main__":
    main()
