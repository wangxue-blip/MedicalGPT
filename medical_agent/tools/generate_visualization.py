# -*- coding: utf-8 -*-
"""generate_visualization：把 findings 的框与掩码画到原图上。

纯工程工具，但有一条纪律：**只画 findings 里有的东西**。它不推断、不补充，
图上出现的每个框都能在 StructuredReport 里找到同一条 bbox——可视化和文字报告
用的是同一份 grounded 数据。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from medical_agent import config
from medical_agent.schema import PATHOLOGY_ZH, TOOL_VISUALIZE, XRV_PATHOLOGIES, BBox
from medical_agent.tools.base import BaseTool

PALETTE = [
    (230, 57, 70), (29, 158, 117), (69, 123, 157), (241, 143, 1), (131, 56, 236),
    (0, 150, 199), (214, 40, 40), (56, 176, 0), (255, 183, 3), (120, 0, 200),
    (2, 62, 138), (247, 127, 0), (0, 128, 128), (188, 71, 73), (77, 144, 142),
    (255, 99, 71), (72, 61, 139), (160, 82, 45),
]

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _color_for(label: str):
    try:
        return PALETTE[XRV_PATHOLOGIES.index(label) % len(PALETTE)]
    except ValueError:
        return PALETTE[hash(label) % len(PALETTE)]


def _load_font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


class GenerateVisualizationTool(BaseTool):
    name = TOOL_VISUALIZE
    description = "在原图上叠加 bbox / 掩码 / 标签，生成标注图"

    def __init__(self, mock: bool = False, output_dir: str = config.OUTPUT_DIR):
        super(GenerateVisualizationTool, self).__init__(mock=mock)
        self.output_dir = output_dir

    def _probe(self) -> str:
        from PIL import Image, ImageDraw  # noqa: F401

        return "pillow"

    def _run_real(self, image_path: str, findings: List[Dict[str, Any]], heatmap_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        from PIL import Image, ImageDraw

        base = Image.open(image_path).convert("RGBA")
        width, height = base.size
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _load_font(max(12, width // 50))
        drawn = 0

        for finding in findings or []:
            label = str(finding.get("label", ""))
            color = _color_for(label)
            mask_info = finding.get("mask") or {}
            mask_path = mask_info.get("path") if isinstance(mask_info, dict) else ""
            if mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert("L").resize(base.size)
                    tint = Image.new("RGBA", base.size, color + (90,))
                    overlay.paste(tint, (0, 0), mask)
                except Exception:
                    pass
            box = finding.get("bbox")
            if box:
                try:
                    x1, y1, x2, y2 = BBox.from_list(box).to_pixels(width, height)
                except ValueError:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=max(2, width // 300))
                score = finding.get("score", 0.0)
                text = "%s %d%%" % (PATHOLOGY_ZH.get(label, label), int(round(float(score) * 100)))
                text_box = draw.textbbox((x1, max(0, y1 - font.size - 4)), text, font=font)
                draw.rectangle(text_box, fill=color + (200,))
                draw.text((x1, max(0, y1 - font.size - 4)), text, fill=(255, 255, 255, 255), font=font)
                drawn += 1

        if drawn == 0:
            draw.text((10, 10), "未检出异常区域", fill=(255, 255, 255, 255), font=font)

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        out_dir = Path(self.output_dir) / "vis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / ("%s_annotated.png" % Path(image_path).stem))
        composed.save(out_path)
        return {"visualization_path": out_path, "boxes_drawn": drawn}

    def _run_mock(self, image_path: str, findings: List[Dict[str, Any]], heatmap_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {"visualization_path": "", "boxes_drawn": len([f for f in (findings or []) if f.get("bbox")])}
