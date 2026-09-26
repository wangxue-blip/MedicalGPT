# -*- coding: utf-8 -*-
"""medical_agent 的五个工具。具体实现见各模块；图节点只通过 ToolRegistry 访问。"""

from medical_agent.tools.base import BaseTool, ToolRegistry

__all__ = ["BaseTool", "ToolRegistry"]
