# -*- coding: utf-8 -*-
"""detect_abnormality：胸片异常候选分类。

用 torchxrayvision 的 DenseNet121（在 CheXpert / NIH / MIMIC 等多个胸片数据集上
预训练，18 类病理）做多标签分类，不训练。方案里明确说了 COCO 预训练的 YOLO
不能直接识别医学病灶——它的特征是为彩色自然图像学的，X 光是灰度投影图。

输出是"候选异常类别 + 模型分数"，不是诊断。分数低于 ``min_score`` 的不作为
finding，避免把 0.1 的噪声当发现写进报告。``preprocess`` 与 ``model`` 公开，供
localize_finding 复用同一个网络做 Grad-CAM，不再加载第二份权重。
"""

from typing import Any, Dict, Tuple

from medical_agent import config
from medical_agent.schema import TOOL_DETECT, XRV_PATHOLOGIES, Finding
from medical_agent.tools.base import BaseTool


class DetectAbnormalityTool(BaseTool):
    name = TOOL_DETECT
    description = "胸片多标签异常分类（torchxrayvision DenseNet121，18 类病理）"

    def __init__(
        self,
        mock: bool = False,
        weights: str = config.XRV_WEIGHTS,
        topk: int = config.DETECT_TOPK,
        min_score: float = config.DETECT_MIN_SCORE,
        image_size: int = config.IMAGE_SIZE,
        device: str = "cpu",
    ):
        super(DetectAbnormalityTool, self).__init__(mock=mock, device=device)
        self.weights = weights or "densenet121-res224-all"
        self.topk = max(1, int(topk or 1))
        self.min_score = float(min_score)
        self.image_size = int(image_size or 224)
        self.model = None
        self._torch = None
        self._xrv = None

    # -- 真实后端 ------------------------------------------------------------
    def _probe(self) -> str:
        import torch
        import torchxrayvision as xrv

        self._torch = torch
        self._xrv = xrv
        model = xrv.models.DenseNet(weights=self.weights)
        model.eval()
        model.to(self.device)
        self.model = model
        return "torchxrayvision:%s" % self.weights

    def preprocess(self, image_path: str) -> Tuple[Any, Tuple[int, int]]:
        """PIL 灰度 -> xrv 归一化(-1024~1024) -> 中心裁方 -> 缩放到 image_size。

        返回 ``(tensor[1,1,S,S], (orig_w, orig_h))``。缩放用 torch 自己做，
        避免依赖 skimage。
        """
        import numpy as np
        from PIL import Image

        torch = self._torch
        image = Image.open(image_path).convert("L")
        width, height = image.size
        array = np.asarray(image, dtype=np.float32)
        array = self._xrv.datasets.normalize(array, 255)
        side = min(array.shape[0], array.shape[1])
        top = (array.shape[0] - side) // 2
        left = (array.shape[1] - side) // 2
        array = array[top:top + side, left:left + side]
        tensor = torch.from_numpy(array)[None, None, :, :]
        tensor = torch.nn.functional.interpolate(
            tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False,
        )
        return tensor, (width, height)

    def _run_real(self, image_path: str) -> Dict[str, Any]:
        torch = self._torch
        tensor, (width, height) = self.preprocess(image_path)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
        probs = output[0].detach().float().cpu()
        # xrv 的 DenseNet forward 已做 sigmoid；保险起见越界时再做一次
        if float(probs.min()) < 0.0 or float(probs.max()) > 1.0:
            probs = torch.sigmoid(probs)
        labels = list(self.model.pathologies)
        all_scores = {}
        for label, prob in zip(labels, probs.tolist()):
            if label:
                all_scores[label] = round(float(prob), 6)
        ranked = sorted(
            [(label, score) for label, score in all_scores.items() if label in XRV_PATHOLOGIES and score >= self.min_score],
            key=lambda item: item[1], reverse=True,
        )[:self.topk]
        findings = [Finding(label=label, score=score, source_tool=TOOL_DETECT).to_dict() for label, score in ranked]
        return {"findings": findings, "all_scores": all_scores, "image_size": [width, height]}

    # -- mock ------------------------------------------------------------------
    MOCK_SCORES = (("Lung Opacity", 0.82), ("Cardiomegaly", 0.41))

    def _run_mock(self, image_path: str) -> Dict[str, Any]:
        findings = [
            Finding(label=label, score=score, source_tool=TOOL_DETECT).to_dict()
            for label, score in self.MOCK_SCORES
            if score >= self.min_score
        ][:self.topk]
        return {
            "findings": findings,
            "all_scores": {label: score for label, score in self.MOCK_SCORES},
            "image_size": [1024, 1024],
        }
