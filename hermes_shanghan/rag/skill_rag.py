"""Skill RAG — selects and invokes the correct Shanghan skill.

Pipeline (per protocol):
  用戶問題 → 判斷用戶角色 → 檢索 Skill → 調用對應規則 → 回源原文
  → 生成答案 → 安全審查

Routing combines intent keywords (鑒別/誤治/禁忌/條文/論文/通俗…) with BM25
over compiled SKILL.md documents. Patient role always lands in the
patient_education skill with the intent guard armed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .. import config, lexicon, safety
from ..textutil import normalize_query
from .bm25 import BM25Index

INTENT_ROUTES = [
    # (regex on normalized query, skill name, handler key)
    (r"(鑒別|鉴别|區別|区别|怎麼分|怎么分|不同|vs|對比|对比)", "hermes.shanghan.differential", "differential"),
    (r"(誤治|误治|誤下|误下|誤汗|误汗|火逆|燒針|烧针|壞病|坏病|傳變|传变|變證|变证)", "hermes.shanghan.mistreatment", "mistreatment"),
    (r"(禁忌|不可發汗|不可汗|不可下|不可吐|禁汗|禁下|禁吐|禁例)", "hermes.shanghan.contraindications", "contraindication"),
    (r"(第?\d+[條条]|條文|条文|解釋.*條|解释.*条|原文)", "hermes.shanghan.clause_explainer", "clause"),
    (r"(論文|论文|paper|投稿|寫作|写作|摘要|圖表|图表)", "hermes.shanghan.paper_writer", "paper"),
    (r"(通俗|什麼意思|什么意思|聽不懂|听不懂|解釋給我|解释给我|患者|病人)", "hermes.shanghan.patient_education", "patient"),
    (r"(治法|汗法|下法|吐法|和法|溫法|温法|清法|救逆)", "hermes.shanghan.therapy", "therapy"),
    (r"(太陽病|阳明病|陽明病|少陽病|少阳病|太陰病|太阴病|少陰病|少阴病|厥陰病|厥阴病|六經|六经|提綱|提纲)", "hermes.shanghan.six_channels", "six_channel"),
]


class SkillRAG:
    def __init__(self, skills_root: Optional[Path] = None):
        self.root = skills_root or config.SKILLS_DIR
        self.skills: Dict[str, Dict] = {}
        self.index = BM25Index()
        self._load()

    def _load(self):
        for md in sorted(self.root.rglob("SKILL.md")):
            text = md.read_text(encoding="utf-8")
            m = re.search(r"^name:\s*(.+)$", text, re.M)
            name = m.group(1).strip() if m else md.parent.name
            dm = re.search(r"^description:\s*(.+)$", text, re.M)
            desc = dm.group(1).strip() if dm else ""
            rules_path = md.parent / "rules.jsonl"
            self.skills[name] = {
                "name": name, "dir": str(md.parent), "description": desc,
                "skill_md": text, "rules_path": str(rules_path),
            }
            self.index.add(name, desc + "\n" + text[:2000])
        self.index.finalize()

    # ------------------------------------------------------------------
    def infer_role(self, question: str, role: Optional[str] = None) -> str:
        if role in safety.ROLES:
            return role
        q = question
        # conservative default: explicit prescription/dosage seeking without a
        # declared professional role is treated as patient → intent guard
        if safety.RE_PRESCRIPTION_INTENT.search(q) or safety.RE_DOSAGE_INTENT.search(q) \
           or safety.RE_DIAGNOSIS_INTENT.search(q):
            return "patient"
        if re.search(r"(我|医生说|醫生說|爸|媽|妈|家人)", q) and \
           re.search(r"(什麼意思|什么意思|怎麼回事|怎么回事|要紧|要緊|严重|嚴重)", q):
            return "patient"
        if re.search(r"(論文|论文|paper|數據|数据|挖掘|圖譜|图谱|網絡|网络|統計|统计)", q):
            return "researcher"
        if re.search(r"(學習|学习|考試|考试|背|講解|讲解|教學|教学|練習|练习)", q):
            return "student"
        return "doctor"

    def route(self, question: str, role: Optional[str] = None) -> Dict:
        q = normalize_query(question)
        role = self.infer_role(question, role)

        # patient role: always patient_education skill
        if role == "patient":
            return {"role": role, "skill": "hermes.shanghan.patient_education",
                    "handler": "patient", "match": "role_forced"}

        for pattern, skill, handler in INTENT_ROUTES:
            if re.search(pattern, q) or re.search(pattern, question):
                # formula mention upgrades differential/six_channel to formula skill
                if handler in ("six_channel",):
                    channel = next((config.CHANNEL_PINYIN[c] for c in config.CHANNEL_PINYIN
                                    if c in q), None)
                    if channel:
                        return {"role": role,
                                "skill": f"hermes.shanghan.{channel}",
                                "handler": "six_channel", "match": f"intent:{pattern}"}
                return {"role": role, "skill": skill, "handler": handler,
                        "match": f"intent:{pattern}"}

        # formula-name routing
        for name in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True):
            if name in q:
                from ..skills.pinyin import formula_slug
                cand = f"hermes.formula.{formula_slug(name)}"
                if cand in self.skills:
                    return {"role": role, "skill": cand, "handler": "formula",
                            "match": f"formula:{name}"}

        # symptom-list questions → doctor matching
        if re.search(r"(惡寒|恶寒|發熱|发热|脈|脉|無汗|无汗|汗出)", q) and role == "doctor":
            return {"role": role, "skill": "hermes.shanghan.differential",
                    "handler": "match", "match": "findings"}

        hits = self.index.search(q, top_k=1)
        if hits:
            return {"role": role, "skill": hits[0][0], "handler": "generic",
                    "match": f"bm25:{hits[0][1]}"}
        return {"role": role, "skill": "hermes.shanghan.catalog",
                "handler": "generic", "match": "fallback"}

    # ------------------------------------------------------------------
    def skill_rules(self, skill_name: str, limit: int = 30) -> List[Dict]:
        info = self.skills.get(skill_name)
        if not info:
            return []
        path = Path(info["rules_path"])
        out = []
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= limit:
                        break
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out

    def describe(self) -> List[Dict]:
        return [{"name": k, "description": v["description"]}
                for k, v in sorted(self.skills.items())]
