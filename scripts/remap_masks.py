import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap target masks to binary damage masks.")
    parser.add_argument("--input-dir", default="dataset/test/target")
    parser.add_argument("--output-dir", default="dataset/test/binary_target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(input_dir.glob("*.tif"))
    if not mask_paths:
        raise ValueError(f"No .tif masks found in {input_dir}")

    for mask_path in tqdm(mask_paths):
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask >= 2).astype(np.uint8)
        Image.fromarray(binary).save(output_dir / mask_path.name)

    print(f"Wrote {len(mask_paths)} binary masks to {output_dir}")


if __name__ == "__main__":
    main()
