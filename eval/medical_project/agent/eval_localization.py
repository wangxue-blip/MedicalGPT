#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""病灶定位评测：bbox IoU 与 mask Dice，对照 CheXlocalize 专家标注。

gt 与 pred 均为 jsonl，每行 ``{"image_id", "label", "bbox": [x1,y1,x2,y2 归一化], "mask_rle"?, "mask_size"?}``。
按 (image_id, label) 配对；gt 没有对应预测时该样本记 0（漏检也要算进去，否则
只对"检出的"算 IoU 会虚高）。

Dice 只在双方都有 RLE 且 numpy 可用时计算。COCO 非压缩 RLE 是**列优先**
（Fortran order），解码时必须按这个顺序 reshape，否则掩码整体转置。

这些是启发式自动指标，量化的是"CAM 粗定位与专家标注差多少"，不等于临床正确性。
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.schema import BBox, read_jsonl  # noqa: E402


def decode_rle(counts, size):
    """COCO 非压缩 RLE -> 二值掩码 (numpy)，列优先。counts 交替为 0 段与 1 段长度。"""
    import numpy as np

    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    value = 0
    for run in counts:
        run = int(run)
        if value:
            flat[position:position + run] = 1
        position += run
        value = 1 - value
    return flat.reshape((width, height)).T  # Fortran order


def dice(mask_a, mask_b):
    inter = float((mask_a & mask_b).sum())
    total = float(mask_a.sum() + mask_b.sum())
    return 2.0 * inter / total if total > 0 else 0.0


def _safe_bbox(values):
    try:
        return BBox.from_list(values) if values else None
    except ValueError:
        return None


def evaluate(gt_rows, pred_rows, iou_threshold=0.5):
    preds = {(r.get("image_id"), r.get("label")): r for r in pred_rows}
    per_label = defaultdict(lambda: {"iou": [], "dice": [], "hit": []})
    dice_available = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        dice_available = False

    for gt in gt_rows:
        key = (gt.get("image_id"), gt.get("label"))
        bucket = per_label[gt.get("label", "")]
        pred = preds.get(key)
        gt_box = _safe_bbox(gt.get("bbox"))
        pred_box = _safe_bbox(pred.get("bbox")) if pred else None
        iou = gt_box.iou(pred_box) if (gt_box and pred_box) else 0.0
        bucket["iou"].append(iou)
        bucket["hit"].append(float(iou >= iou_threshold))
        if dice_available and pred and gt.get("mask_rle") and pred.get("mask_rle") and gt.get("mask_size"):
            try:
                gt_mask = decode_rle(gt["mask_rle"], gt["mask_size"])
                pred_mask = decode_rle(pred["mask_rle"], pred.get("mask_size") or gt["mask_size"])
                if gt_mask.shape == pred_mask.shape:
                    bucket["dice"].append(dice(gt_mask.astype(bool), pred_mask.astype(bool)))
            except Exception:
                pass

    labels_out = {}
    all_iou, all_dice, all_hit = [], [], []
    for label, bucket in sorted(per_label.items()):
        labels_out[label] = {
            "n": len(bucket["iou"]),
            "mean_iou": statistics.mean(bucket["iou"]) if bucket["iou"] else 0.0,
            "hit_rate": statistics.mean(bucket["hit"]) if bucket["hit"] else 0.0,
            "mean_dice": statistics.mean(bucket["dice"]) if bucket["dice"] else None,
        }
        all_iou.extend(bucket["iou"])
        all_dice.extend(bucket["dice"])
        all_hit.extend(bucket["hit"])
    return {
        "n_gt": len(gt_rows),
        "n_pred": len(pred_rows),
        "iou_threshold": iou_threshold,
        "mean_iou": statistics.mean(all_iou) if all_iou else 0.0,
        "hit_rate": statistics.mean(all_hit) if all_hit else 0.0,
        "mean_dice": statistics.mean(all_dice) if all_dice else None,
        "dice_available": dice_available,
        "per_label": labels_out,
    }


def main():
    parser = argparse.ArgumentParser(description="病灶定位评测（IoU / Dice / hit@IoU）")
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gradcam")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = {"model": args.model, "gt_file": args.gt_file, "pred_file": args.pred_file}
    result.update(evaluate(list(read_jsonl(args.gt_file)), list(read_jsonl(args.pred_file)), args.iou_threshold))
    result["note"] = "IoU/Dice 是与专家标注的几何重合度，不等于临床正确性；漏检样本按 0 计入。"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
