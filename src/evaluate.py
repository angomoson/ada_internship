import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted binary masks against target masks."
    )

    parser.add_argument(
        "--pred-dir",
        default="outputs/test_predictions"
    )

    parser.add_argument(
        "--target-dir",
        default="dataset/test/target"
    )

    parser.add_argument(
        "--save-dir",
        default="evaluation_results",
        help="Directory to save plots and outputs."
    )

    parser.add_argument(
        "--remap-target",
        action="store_true",
        default=True,
        help="Map target labels 0,1 -> 0 and 2,3 -> 1 before evaluation.",
    )

    return parser.parse_args()


def binary_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def plot_metrics(scores: dict, save_path: Path):
    metric_names = list(scores.keys())
    metric_values = list(scores.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(metric_names, metric_values)

    plt.ylim(0, 1)

    plt.ylabel("Score")
    plt.title("Segmentation Metrics")

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(tp, fp, fn, tn, save_path: Path):
    cm = np.array([
        [tp, fn],
        [fp, tn]
    ])

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Pred Positive", "Pred Negative"])
    ax.set_yticklabels(["True Positive", "True Negative"])

    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")

    plt.title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12
            )

    plt.colorbar(im)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    target_dir = Path(args.target_dir)
    save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_dir.glob("*.tif"))

    if not pred_paths:
        raise ValueError(f"No prediction .tif files found in {pred_dir}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    evaluated = 0
    missing_targets: list[str] = []

    per_image_ious = []

    for pred_path in tqdm(pred_paths):
        target_path = target_dir / pred_path.name

        if not target_path.exists():
            missing_targets.append(pred_path.name)
            continue

        pred = np.array(Image.open(pred_path).convert("L"))

        target = np.array(Image.open(target_path).convert("L"))

        if pred.shape != target.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    target.shape[::-1],
                    resample=Image.NEAREST
                )
            )

        pred_bin = pred > 0

        target_bin = (
            target >= 2
            if args.remap_target
            else target > 0
        )

        tp = int(np.logical_and(pred_bin, target_bin).sum())

        fp = int(
            np.logical_and(
                pred_bin,
                np.logical_not(target_bin)
            ).sum()
        )

        fn = int(
            np.logical_and(
                np.logical_not(pred_bin),
                target_bin
            ).sum()
        )

        tn = int(
            np.logical_and(
                np.logical_not(pred_bin),
                np.logical_not(target_bin)
            ).sum()
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        image_iou = (
            tp / (tp + fp + fn)
            if (tp + fp + fn) > 0
            else 1.0
        )

        per_image_ious.append(image_iou)

        evaluated += 1

    if evaluated == 0:
        raise ValueError("No matching prediction/target mask pairs were found.")

    scores = binary_scores(total_tp, total_fp, total_fn)

    print(f"evaluated_images={evaluated}")
    print(f"missing_targets={len(missing_targets)}")

    print(
        f"tp={total_tp} "
        f"fp={total_fp} "
        f"fn={total_fn} "
        f"tn={total_tn}"
    )

    print(f"iou={scores['iou']:.4f}")
    print(f"precision={scores['precision']:.4f}")
    print(f"recall={scores['recall']:.4f}")
    print(f"f1={scores['f1']:.4f}")

    # -----------------------------
    # Save Metrics Bar Chart
    # -----------------------------
    metrics_plot_path = save_dir / "metrics.png"

    plot_metrics(scores, metrics_plot_path)

    print(f"\nSaved metrics chart to:")
    print(metrics_plot_path)

    # -----------------------------
    # Save IoU Distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.hist(per_image_ious, bins=20)

    plt.xlabel("IoU")
    plt.ylabel("Number of Images")
    plt.title("Per-Image IoU Distribution")

    iou_hist_path = save_dir / "iou_distribution.png"

    plt.tight_layout()
    plt.savefig(iou_hist_path)
    plt.close()

    print(f"Saved IoU distribution chart to:")
    print(iou_hist_path)

    # -----------------------------
    # Save Confusion Matrix
    # -----------------------------
    cm_path = save_dir / "confusion_matrix.png"

    plot_confusion_matrix(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
        cm_path
    )

    print(f"Saved confusion matrix to:")
    print(cm_path)

    if missing_targets:
        print("\nFirst missing targets:")

        for name in missing_targets[:10]:
            print(f"  {name}")


if __name__ == "__main__":
    main()