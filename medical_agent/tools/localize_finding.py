# -*- coding: utf-8 -*-
"""localize_finding：用 Grad-CAM 给检测到的病理做可解释定位，同时输出框与掩码。

为什么是 Grad-CAM 而不是目标检测器：第一阶段不训练任何模型。detect_abnormality
已经有一个在胸片上预训练的分类网络，Grad-CAM 从它的最后一个卷积块直接导出
"网络看哪里做的判断"，零额外权重。代价是 CAM 是注意力图不是精确分割，边界
粗糙——这一点在 ``extra["localized_by"]="gradcam"`` 里显式标出，评测时和
CheXlocalize 的专家标注比 IoU/Dice 就能量化差多少。

为什么同时给 bbox 和 mask：弥漫性阴影、间质改变不是紧凑物体，矩形框表示不了；
但下游可视化与 IoU 评测又需要框。所以两者都给，mask 存 PNG 路径不内联像素。

Grad-CAM 手工实现（forward hook 抓特征图、backward hook 抓梯度），不依赖
pytorch-grad-cam。每个 label 单独前向 + 反传，hook 在函数退出时一定移除。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medical_agent import config
from medical_agent.schema import TOOL_LOCALIZE, BBox, Finding, Mask
from medical_agent.tools.base import BaseTool

MOCK_BOXES = {
    "Lung Opacity": [0.10, 0.45, 0.48, 0.90],
    "Cardiomegaly": [0.30, 0.40, 0.72, 0.85],
}
MOCK_DEFAULT_BOX = [0.25, 0.25, 0.75, 0.75]


def _foreground_bbox(mask) -> Optional[Tuple[int, int, int, int]]:
    """最大连通域的外接框（像素 xyxy）。scipy 缺失时退化为整个前景的外接框。

    阈值化后的 CAM 通常只有一个主血块，整体外接框与最大连通域相差不大；真正
    多灶的情况会在评测里以 IoU 偏低的形式暴露出来，不在这里静默处理。
    """
    import numpy as np

    if not mask.any():
        return None
    try:
        from scipy import ndimage

        labeled, count = ndimage.label(mask)
        if count > 1:
            sizes = ndimage.sum(mask, labeled, range(1, count + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = labeled == largest
    except Exception:
        pass
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class LocalizeFindingTool(BaseTool):
    name = TOOL_LOCALIZE
    description = "Grad-CAM 病理定位，输出归一化 bbox 与掩码 PNG"

    def __init__(
        self,
        mock: bool = False,
        detector=None,
        cam_threshold: float = config.CAM_THRESHOLD,
        min_box_area: float = config.MIN_BOX_AREA,
        max_mask_area: float = config.MAX_MASK_AREA,
        output_dir: str = config.OUTPUT_DIR,
        device: str = "cpu",
    ):
        super(LocalizeFindingTool, self).__init__(mock=mock, device=device)
        self.detector = detector
        self.cam_threshold = float(cam_threshold)
        self.min_box_area = float(min_box_area)
        self.max_mask_area = float(max_mask_area)
        self.output_dir = output_dir

    # -- 真实后端 ------------------------------------------------------------
    def _probe(self) -> str:
        if self.detector is None:
            raise RuntimeError("localize_finding 需要传入 DetectAbnormalityTool 实例复用其模型")
        if not self.detector.available:
            raise RuntimeError("检测器不可用: %s" % self.detector.unavailable_reason)
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401

        return "gradcam:%s" % self.detector.backend

    def _run_real(self, image_path: str, labels: List[str]) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        model = self.detector.model
        tensor, (width, height) = self.detector.preprocess(image_path)
        tensor = tensor.to(self.device)
        pathologies = list(model.pathologies)
        captured: Dict[str, Any] = {}

        def forward_hook(_module, _inputs, output):
            captured["activation"] = output

        def backward_hook(_module, _grad_input, grad_output):
            captured["gradient"] = grad_output[0]

        target_layer = model.features
        handles = [
            target_layer.register_forward_hook(forward_hook),
            target_layer.register_full_backward_hook(backward_hook),
        ]
        out_dir = Path(self.output_dir) / "localize"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        findings: List[Dict[str, Any]] = []
        heatmap_paths: Dict[str, str] = {}
        skipped: Dict[str, str] = {}
        try:
            for label in labels or []:
                if label not in pathologies:
                    skipped[label] = "not_in_model"
                    continue
                index = pathologies.index(label)
                model.zero_grad()
                captured.clear()
                output = model(tensor)
                score_tensor = output[0, index]
                score_tensor.backward()
                if "activation" not in captured or "gradient" not in captured:
                    skipped[label] = "hook_missed"
                    continue
                activation = captured["activation"][0].detach()      # [C,h,w]
                gradient = captured["gradient"][0].detach()          # [C,h,w]
                weights = gradient.mean(dim=(1, 2))                  # [C]
                cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
                cam_max = float(cam.max())
                if cam_max <= 0.0:
                    skipped[label] = "zero_cam"
                    continue
                cam = (cam - cam.min()) / (cam_max - float(cam.min()) + 1e-8)
                cam_up = torch.nn.functional.interpolate(
                    cam[None, None, :, :], size=(height, width), mode="bilinear", align_corners=False,
                )[0, 0].cpu().numpy()
                mask = cam_up >= self.cam_threshold
                area_ratio = float(mask.mean())
                if area_ratio <= 0.0:
                    skipped[label] = "empty_mask"
                    continue
                pixel_box = _foreground_bbox(mask)
                if pixel_box is None:
                    skipped[label] = "no_component"
                    continue
                x1, y1, x2, y2 = pixel_box
                try:
                    bbox = BBox(x1 / float(width), y1 / float(height), min(1.0, x2 / float(width)), min(1.0, y2 / float(height)))
                    bbox.validate()
                except ValueError as exc:
                    skipped[label] = "bad_bbox:%s" % exc
                    continue
                if bbox.area < self.min_box_area:
                    skipped[label] = "tiny_bbox"
                    continue
                safe_label = label.replace(" ", "_").replace("/", "_")
                mask_path = str(out_dir / ("%s_%s_mask.png" % (stem, safe_label)))
                heat_path = str(out_dir / ("%s_%s_cam.png" % (stem, safe_label)))
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
                Image.fromarray((np.clip(cam_up, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(heat_path)
                heatmap_paths[label] = heat_path
                extra = {"localized_by": "gradcam", "cam_threshold": self.cam_threshold}
                if area_ratio > self.max_mask_area:
                    extra["degenerate"] = True
                findings.append(Finding(
                    label=label,
                    score=max(0.0, min(1.0, float(score_tensor.detach()))),
                    source_tool=TOOL_LOCALIZE,
                    bbox=bbox,
                    mask=Mask(path=mask_path, area_ratio=round(area_ratio, 6)),
                    extra=extra,
                ).to_dict())
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad()
        return {"findings": findings, "heatmap_paths": heatmap_paths, "skipped": skipped, "image_size": [width, height]}

    # -- mock ------------------------------------------------------------------
    def _run_mock(self, image_path: str, labels: List[str]) -> Dict[str, Any]:
        findings = []
        for label in labels or []:
            box = MOCK_BOXES.get(label, MOCK_DEFAULT_BOX)
            findings.append(Finding(
                label=label,
                score=0.82 if label == "Lung Opacity" else 0.41 if label == "Cardiomegaly" else 0.5,
                source_tool=TOOL_LOCALIZE,
                bbox=BBox.from_list(box),
                mask=Mask(path="", area_ratio=0.15),
                extra={"localized_by": "mock"},
            ).to_dict())
        return {"findings": findings, "heatmap_paths": {}, "skipped": {}, "image_size": [1024, 1024]}
