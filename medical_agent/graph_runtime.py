# -*- coding: utf-8 -*-
"""图运行时：优先 LangGraph，缺失时用一个语义等价的最小回退。

为什么要回退实现而不是硬依赖 LangGraph：
* 单测与冒烟必须能在裸环境跑通，这是本仓库所有子系统的既定纪律；
* 回退运行时只实现 LangGraph 中本项目用到的子集（add_node / add_edge /
  add_conditional_edges / set_entry_point / compile / invoke），语义对齐：
  节点返回的 dict 合并进 state，条件边由路由函数返回的键经映射表决定下一节点。

真实部署时装上 langgraph，本模块自动切换，上层代码零改动。用
``HAS_LANGGRAPH`` 判断当前走的是哪条路，并把它写进运行输出——降级不能被悄悄略过。
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    HAS_LANGGRAPH = True
    RUNTIME_BACKEND = "langgraph"
except Exception as exc:  # ImportError 或版本不兼容
    HAS_LANGGRAPH = False
    RUNTIME_BACKEND = "fallback"
    logger.warning("langgraph 不可用(%s)，使用内置回退运行时", exc)

    START = "__start__"
    END = "__end__"

    class _CompiledGraph(object):
        def __init__(self, nodes, edges, conditional, entry, max_steps):
            self._nodes = nodes
            self._edges = edges
            self._conditional = conditional
            self._entry = entry
            self._max_steps = max_steps

        def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """与 LangGraph 一致：节点返回的 dict 浅合并进 state，直到到达 END。"""
            state = dict(state)
            current = self._entry
            steps = 0
            while current != END:
                if current not in self._nodes:
                    raise KeyError("未注册的节点: %r" % current)
                steps += 1
                if steps > self._max_steps:
                    raise RuntimeError("图执行超过 %d 步，疑似死循环，最后节点 %r" % (self._max_steps, current))
                update = self._nodes[current](state)
                if update:
                    state.update(update)
                if current in self._conditional:
                    router, mapping = self._conditional[current]
                    key = router(state)
                    current = mapping.get(key, key) if mapping else key
                elif current in self._edges:
                    current = self._edges[current]
                else:
                    current = END
            return state

    class StateGraph(object):  # noqa: D101 - 与 langgraph 同名，故意
        def __init__(self, state_schema=None, max_steps: int = 64):
            self._nodes = {}
            self._edges = {}
            self._conditional = {}
            self._entry = None
            self._max_steps = max_steps

        def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
            if name in (START, END):
                raise ValueError("节点名不能是 %r" % name)
            self._nodes[name] = fn
            return self

        def add_edge(self, src: str, dst: str):
            if src == START:
                self._entry = dst
            else:
                self._edges[src] = dst
            return self

        def add_conditional_edges(self, src: str, router: Callable[[Dict[str, Any]], str], mapping: Optional[Dict[str, str]] = None):
            self._conditional[src] = (router, dict(mapping or {}))
            return self

        def set_entry_point(self, name: str):
            self._entry = name
            return self

        def compile(self, **kwargs):
            if self._entry is None:
                raise ValueError("未设置入口节点")
            return _CompiledGraph(self._nodes, self._edges, self._conditional, self._entry, self._max_steps)


__all__ = ["StateGraph", "START", "END", "HAS_LANGGRAPH", "RUNTIME_BACKEND"]
