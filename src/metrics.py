import torch


def binary_confusion_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    return int(tp), int(fp), int(fn)


def binary_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def binary_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= threshold)
    truth = (target >= 0.5)
    intersection = (pred & truth).sum().item()
    union = (pred | truth).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def multiclass_mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> float:
    pred = torch.argmax(logits, dim=1)
    ious: list[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        intersection = (pred_cls & target_cls).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 1.0
