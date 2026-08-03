"""文档质量诊断：Router 前置的静态分析，产出 DocQualityReport 供路由与展示。"""

from .md_diagnosis import DocQualityReport, diagnose

__all__ = ["DocQualityReport", "diagnose"]
