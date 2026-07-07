"""Agentic layer: tool registry, citation guard, and a provider-agnostic
tool-calling agent that keeps every answer leashed to clause evidence.

Enhanced stack (任務規劃／證據綁定／多假設推理／合議裁決)：
  Planner           複合問題 → 帶依賴的任務圖 + success_criteria
  EvidenceBinder    回答逐句綁定本輪證據（claim→clause_id→層級→置信）
  HypothesisManager 多假設方證分析 + 鑒別追問（ClarificationAgent）
  ConsensusJudge    多專家合議的共識/分歧/需補充裁決
"""
from .agent import ShanghanAgent
from .consensus import ConsensusJudge
from .evidence_binder import EvidenceBinder
from .hypothesis import HypothesisManager
from .planner import Planner
from .tools import PATIENT_SAFE_TOOLS, ToolRegistry, get_registry

__all__ = ["ShanghanAgent", "ToolRegistry", "get_registry",
           "Planner", "EvidenceBinder", "HypothesisManager",
           "ConsensusJudge", "PATIENT_SAFE_TOOLS"]
