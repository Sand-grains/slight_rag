"""Agent 编排层：基于 hello-agents 框架的 RAG 搜索工具。

公共接口：
    - RAGSearchTool: hello-agents Tool 子类，将 Retriever 暴露为 Agent 可调用工具
"""

from .tools import RAGSearchTool
