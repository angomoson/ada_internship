from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


Task = Literal["binary", "multiclass"]


def remap_damage_mask(mask: torch.Tensor) -> torch.Tensor:
    """Map original labels 0,1 to background and 2,3 to damaged foreground."""
    return (mask >= 2).float()


class DamageMaskDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        names: list[str] | None = None,
        image_size: int = 512,
        task: Task = "binary",
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.pre_dir = self.data_root / "pre-event"
        self.post_dir = self.data_root / "post-event"
        self.mask_dir = self.data_root / "target"
        self.image_size = image_size
        self.task = task
        self.augment = augment

        if names is None:
            names = sorted(p.name for p in self.pre_dir.glob("*.tif"))

        self.names = [
            name
            for name in names
            if (self.pre_dir / name).exists()
            and (self.post_dir / name).exists()
            and (self.mask_dir / name).exists()
        ]

        if not self.names:
            raise ValueError(f"No matched TIFF triplets found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.names[idx]

        pre = np.array(Image.open(self.pre_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        post = np.array(Image.open(self.post_dir / name).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_dir / name).convert("L"), dtype=np.int64)

        x = np.concatenate([pre.transpose(2, 0, 1), post[None, ...]], axis=0)
        image = torch.from_numpy(x)
        target = torch.from_numpy(mask)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        target = F.interpolate(
            target.float().unsqueeze(0).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        if self.task == "binary":
            target = remap_damage_mask(target).unsqueeze(0)
        else:
            target = target.clamp(min=0, max=3)

        if self.augment:
            image, target = self._augment(image, target)

        return {"image": image, "mask": target, "name": name}

    @staticmethod
    def _augment(
        image: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[2])
            target = torch.flip(target, dims=[-1])
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[1])
            target = torch.flip(target, dims=[-2])
        if torch.rand(()) < 0.5:
            k = int(torch.randint(1, 4, ()).item())
            image = torch.rot90(image, k, dims=[1, 2])
            target = torch.rot90(target, k, dims=[-2, -1])
        return image, target
