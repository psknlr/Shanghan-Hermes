"""Agent 執行 harness：顯式狀態圖 · checkpoint/resume/replay · span 級軌跡 ·
發布閘門（含人工審核節點）。

把「隱式 while-loop 編排」升級為可恢復、可觀測、可審計的運行時：

    RunSpec + StateGraph + TraceStore + EvidenceLedger
    + CitationGuard + ReleaseGate(含 HumanReviewGate) + Checkpoint

設計約束（與全庫一致）：純標準庫（無 Pydantic/OTel SDK，實現兼容結構）；
運行目錄 `data/shanghan/runs/<run_id>/`（state.json + events.jsonl，含時間戳
故不入庫）；local 後端下 replay 全確定。
"""
from .runner import HarnessRunner, load_run, list_runs  # noqa: F401
from .state import NodeSpec, RunSpec, RunState          # noqa: F401
