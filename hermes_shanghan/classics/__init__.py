"""classics — 獨立於傷寒論領域的全量古籍智能體子系統（十五輪）。

平台級能力（不帶 shanghan 前綴、不依賴傷寒論規則庫）：

- model      通用 Passage/Span 身份模型（work/witness/section/passage/span，
             sha256 穩定 ID）
- search     分層檢索（L0 元數據篩選 → L1 字符倒排剪枝 → L2 逐字驗證，
             布爾/鄰近/命中座標/全量計數，每層可解釋）
- evidence   P 層 EvidenceRecord（verbatim_text + 字符座標 + quote_hash，
             可重驗）、證據包導出、按結論類型的最低證據層策略
- tools      classics_* 工具族（統一註冊進 ToolRegistry → Broker/MCP/規格）
- agent      ClassicsAgent——第二套智能體：檢索計劃/已查書目/反證/
             首現候選/待人工核驗 全程留痕
- audit      全庫驗收審計（解析率/編碼異常/元數據缺失/嵌套/重複/抽樣金標準）
"""
from .agent import ClassicsAgent            # noqa: F401
from .model import RE_PASSAGE_ID, stable_id  # noqa: F401
