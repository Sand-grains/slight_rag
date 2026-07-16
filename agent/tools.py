from typing import Dict, Any, List
from hello_agents.tools.base import Tool, ToolParameter
from retrieval.retriever import Retriever
from retrieval.generator import _build_context


class RAGSearchTool(Tool):
    """将检索层的 Retriever 封装为 Agent 可调用的工具"""

    def __init__(self, retriever: Retriever):
        super().__init__(
            name="search_knowledge_base",                                        # Agent后续通过此名称匹配工具, 对应 [TOOL_CALL:search_knowledge_base:...]
            description="当回答用户问题之前需要了解一定的背景知识时, 调用此工具在知识库中检索与用户问题相关的文档片段, "
                        "后续整合并返回上下文文本(已标注来源)"  # Agent根据description判断何时调用此工具
        )
        self.retriever = retriever                                               # 持有检索层引用, 不在Tool内部直接操作VectorStore

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",                                                     # 参数名, Agent填query=xxx
                type="string",
                description="要在知识库中检索的问题或某关键词",
                required=True
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        """委托Retriever执行检索，工具只负责格式化结果"""
        query = parameters.get("query", "")
        chunks = self.retriever.retrieve(query)                                   # 检索委托给 Retriever（内部处理向量化+相似度计算）
        if not chunks:
            return "知识库中未检索到相关文档片段"
        return _build_context(chunks)                                             # 格式化为带来源标记的上下文字符串
