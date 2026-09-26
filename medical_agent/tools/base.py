# -*- coding: utf-8 -*-
"""所有工具的父类与注册表。

三条纪律，每个工具都必须遵守：

1. ``run()`` 永不抛异常。任何失败都变成 ``ToolResult(ok=False, error=...)``，
   由图的 aggregate 节点决定怎么降级——工具自己不做业务决策。
2. **真实后端不可用时绝不静默换 mock。** 只有调用方显式 ``mock=True`` 才走假
   后端，且结果永远带 ``backend="mock"``。否则返回 ``ok=False``。医疗系统里
   "外观正常地给假结果"比报错危险得多。
3. 后端探测是惰性的：``available`` 第一次访问才真正 import / 加载模型，
   失败原因记进 ``unavailable_reason``，供 ``retrieval_explain`` 一类的诊断
   输出使用。
"""

import logging
import time
from typing import Any, Dict, Optional

from medical_agent.schema import ToolResult

logger = logging.getLogger(__name__)


class BaseTool(object):
    """子类需实现 ``_probe`` / ``_run_real`` / ``_run_mock``，并设置 ``name``。"""

    name = ""
    description = ""

    def __init__(self, mock: bool = False, device: str = "cpu", **kwargs):
        self.mock = bool(mock)
        self.device = device or "cpu"
        self.options = dict(kwargs)
        self._available: Optional[bool] = None
        self._backend = ""
        self.unavailable_reason = ""

    # -- 子类接口 ------------------------------------------------------------
    def _probe(self) -> str:
        """尝试加载真实后端。成功返回 backend 名（非空字符串），失败抛异常。"""
        raise NotImplementedError

    def _run_real(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def _run_mock(self, **kwargs) -> Dict[str, Any]:
        """确定性的假输出：同样输入必须得到同样结果，单测靠这一点断言。"""
        raise NotImplementedError

    # -- 通用实现 ------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self.mock:
            return True
        if self._available is None:
            try:
                self._backend = self._probe() or "real"
                self._available = True
            except Exception as exc:  # 任何加载失败都只记录，不传播
                self._available = False
                self._backend = ""
                self.unavailable_reason = "%s: %s" % (type(exc).__name__, exc)
                logger.warning("工具 %s 真实后端不可用: %s", self.name, self.unavailable_reason)
        return bool(self._available)

    @property
    def backend(self) -> str:
        if self.mock:
            return "mock"
        if self._available is None:
            _ = self.available
        return self._backend

    def run(self, **kwargs) -> ToolResult:
        started = time.perf_counter()
        if self.mock:
            try:
                payload = self._run_mock(**kwargs)
                return ToolResult(
                    tool=self.name, ok=True, backend="mock",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    payload=payload or {}, degraded=True,
                )
            except Exception as exc:
                return ToolResult(
                    tool=self.name, ok=False, backend="mock",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error="mock 后端异常 %s: %s" % (type(exc).__name__, exc), degraded=True,
                )

        if not self.available:
            return ToolResult(
                tool=self.name, ok=False, backend="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="真实后端不可用: %s" % (self.unavailable_reason or "未知原因"),
                degraded=True,
            )

        try:
            payload = self._run_real(**kwargs)
            return ToolResult(
                tool=self.name, ok=True, backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                payload=payload or {},
            )
        except Exception as exc:
            logger.exception("工具 %s 运行失败", self.name)
            return ToolResult(
                tool=self.name, ok=False, backend=self.backend,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="%s: %s" % (type(exc).__name__, exc),
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mock": self.mock,
            "available": self.available,
            "backend": self.backend,
            "unavailable_reason": self.unavailable_reason,
        }


class ToolRegistry(object):
    """按名字管理工具。图节点只通过注册表拿工具，不直接 import 具体类。"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        if not tool.name:
            raise ValueError("工具未设置 name: %r" % tool)
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self):
        return list(self._tools.keys())

    def run(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool=name, ok=False, error="未注册的工具: %s" % name, degraded=True)
        return tool.run(**kwargs)

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: tool.describe() for name, tool in self._tools.items()}
