#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 CheXlocalize 的专家分割标注转成定位评测 gt。

CheXlocalize（Saporta et al. 2022，Stanford AIMI，需在
https://stanfordaimi.azurewebsites.net/ 申请）提供放射科医生对 CheXpert 验证/测试集
十类发现的分割标注，JSON 结构：
    {"<image_id>": {"<Pathology>": {"counts": [...非压缩 RLE...], "size": [H, W]}, ...}, ...}

输出 jsonl：``{"image_id", "image_path", "label", "bbox", "mask_rle", "mask_size", "area_ratio"}``。
bbox 由掩码外接框归一化得到。标签映射到 XRV_PATHOLOGIES：``Pleural Effusion -> Effusion``；
``Support Devices`` 不是病理，跳过并计数。

COCO 非压缩 RLE 是列优先（Fortran order），解码时按 (W, H) reshape 再转置。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.schema import XRV_PATHOLOGIES, write_jsonl  # noqa: E402

LABEL_MAP = {
    "Pleural Effusion": "Effusion",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Lung Lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
}
SKIP_LABELS = ("Support Devices",)


def rle_to_bbox_and_area(counts, size):
    """不依赖 numpy：直接在游程上算外接框与面积（列优先）。"""
    height, width = int(size[0]), int(size[1])
    position, value = 0, 0
    min_x = min_y = None
    max_x = max_y = -1
    foreground = 0
    for run in counts:
        run = int(run)
        if value and run > 0:
            foreground += run
            for flat in (position, position + run - 1):
                x, y = divmod(flat, height)  # 列优先：flat = x * height + y
                min_x = x if min_x is None else min(min_x, x)
                max_x = max(max_x, x)
            # 一段游程可能跨越多列，中间列的 y 覆盖整列
            start_x, start_y = divmod(position, height)
            end_x, end_y = divmod(position + run - 1, height)
            if start_x == end_x:
                lo, hi = start_y, end_y
            else:
                lo, hi = 0, height - 1
            min_y = lo if min_y is None else min(min_y, lo)
            max_y = max(max_y, hi)
        position += run
        value = 1 - value
    if foreground == 0 or min_x is None:
        return None, 0.0
    bbox = [min_x / float(width), min_y / float(height), (max_x + 1) / float(width), (max_y + 1) / float(height)]
    return bbox, foreground / float(height * width)


def main():
    parser = argparse.ArgumentParser(description="CheXlocalize gt_segmentations -> 定位评测 jsonl")
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--output", default="data_processed/medical_project/agent/chexlocalize/gt.jsonl")
    parser.add_argument("--keep_rle", action="store_true", help="保留 RLE 以便算 Dice（文件会变大）")
    args = parser.parse_args()

    data = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    rows, skipped, unknown = [], Counter(), Counter()
    for image_id, per_label in data.items():
        for raw_label, seg in (per_label or {}).items():
            if raw_label in SKIP_LABELS:
                skipped[raw_label] += 1
                continue
            label = LABEL_MAP.get(raw_label, raw_label)
            if label not in XRV_PATHOLOGIES:
                unknown[raw_label] += 1
                continue
            counts, size = seg.get("counts"), seg.get("size")
            if not counts or not size:
                continue
            bbox, area_ratio = rle_to_bbox_and_area(counts, size)
            if bbox is None:
                continue
            row = {
                "image_id": image_id,
                "image_path": str(Path(args.image_dir) / image_id) if args.image_dir else image_id,
                "label": label,
                "bbox": [round(v, 6) for v in bbox],
                "mask_size": [int(size[0]), int(size[1])],
                "area_ratio": round(area_ratio, 6),
            }
            if args.keep_rle:
                row["mask_rle"] = counts
            rows.append(row)

    written = write_jsonl(args.output, rows)
    summary = {
        "source": "CheXlocalize (Stanford AIMI, 需申请)",
        "written": written,
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "skipped_non_pathology": dict(skipped),
        "unknown_labels": dict(unknown),
        "keep_rle": args.keep_rle,
        "note": "bbox 由专家分割掩码外接框得到；评测按患者/图像划分，不得与训练集重叠。",
    }
    Path(args.output).with_name("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
