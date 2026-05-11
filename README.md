# GalaxEye Mask Prediction

Small PyTorch pipeline for training a segmentation model that predicts building-damage masks from paired pre-event and post-event TIFF images.

## Data Layout

The scripts expect this structure:

```text
/Users/oson/Desktop/ada_internship_proj/dataset/train/
  pre-event/*.tif
  post-event/*.tif
  target/*.tif

/Users/oson/Desktop/ada_internship_proj/dataset/val/
  pre-event/*.tif
  post-event/*.tif
  target/*.tif

/Users/oson/Desktop/ada_internship_proj/dataset/test/
  pre-event/*.tif
  post-event/*.tif
  target/*.tif
```

Each sample uses:

- `pre-event`: RGB image, 3 channels
- `post-event`: grayscale image, 1 channel
- `target`: mask labels

By default the training script converts mask labels to binary using `mask >= 2`, so labels `0` and `1` become background `0`, while labels `2` and `3` become damaged foreground `1`.

## Train

From the project folder:

```bash
cd /Users/oson/Desktop/ada_internship_proj
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a tiny smoke test first to verify the code and dataset paths:

```bash
python3 src/train.py \
  --epochs 1 \
  --batch-size 1 \
  --image-size 64 \
  --base-features 4 \
  --limit 2 \
  --output-dir outputs/smoke_test
```

Run one real baseline epoch:

```bash
python3 src/train.py \
  --epochs 1 \
  --batch-size 2 \
  --image-size 256 \
  --base-features 16 \
  --output-dir outputs/baseline_256
```

Train for more epochs:

```bash
python3 src/train.py \
  --train-root /Users/oson/Desktop/ada_internship_proj/dataset/train \
  --val-root /Users/oson/Desktop/ada_internship_proj/dataset/val \
  --epochs 20 \
  --batch-size 2 \
  --image-size 256 \
  --base-features 16 \
  --output-dir outputs/final_256
```

The best checkpoint is saved to:

```text
outputs/final_256/best_model.pt
```

Training prints these metrics for both train and validation:

```text
loss, IoU, precision, recall, F1
```

## Remap Masks

Training remaps masks on the fly, but you can also write remapped binary masks to disk:

```bash
python3 scripts/remap_masks.py \
  --input-dir dataset/test/target \
  --output-dir dataset/test/binary_target
```

## Predict Masks

```bash
python3 src/predict.py \
  --checkpoint outputs/final_256/best_model.pt \
  --data-root /Users/oson/Desktop/ada_internship_proj/dataset/test \
  --output-dir outputs/test_predictions
```

Predictions are written as TIFF masks with the original filenames.

## Notes

- Binary target remap is `0, 1 -> 0` and `2, 3 -> 1`, matching damaged-vs-not-damaged training.
- The model input has 4 channels: 3 pre-event RGB channels plus 1 post-event grayscale channel.
- The code avoids `segmentation-models-pytorch` and `albumentations`, so it can run with the libraries already available in this environment.
