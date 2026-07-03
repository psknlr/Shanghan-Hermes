"""Multi-agent council — specialist decomposition over the grounded toolset.

A single question is handled by a pipeline of role-specialized agents, each
mapping onto the protocol's Agent roster:

  Planner      (EntityExtractor/SkillRAG)  決定調度哪些專家
  Retriever    (ClassicalTextRAG)          取證：檢索條文
  FormulaAnalyst (FormulaPatternAgent)     方證匹配
  DifferentialAnalyst (DifferentialAgent)  方證鑒別
  ChannelAnalyst (SixChannelInducer)       六經定位
  MistreatmentAnalyst (MistreatmentAgent)  誤治傳變
  Critic       (ShanghanCritic+CitationGuard) 對抗審查 + 引用核驗
  Synthesizer  (ConsensusJudge)            合議綜合

The council is grounded first: every specialist works through the read-only
ToolRegistry, so it runs fully offline (deterministic). When an LLM backend is
available, the Synthesizer (and optionally each specialist) adds fluent prose —
but the final answer still passes the citation guard and safety governor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import lexicon, safety
from ..llm.client import LLMClient, get_client
from ..textutil import normalize_query
from .citation_guard import CitationGuard
from .tools import ToolRegistry, get_registry


@dataclass
class CouncilMessage:
    agent: str
    role_cn: str
    action: str                      # plan|retrieve|analyze|critique|synthesize
    content: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"agent": self.agent, "role_cn": self.role_cn, "action": self.action,
                "content": self.content, "evidence_ids": self.evidence_ids,
                "tool_calls": self.tool_calls, "data": self.data}


SPECIALISTS = {
    "FormulaAnalyst": "方證分析師",
    "DifferentialAnalyst": "鑒別診斷師",
    "ChannelAnalyst": "六經定位師",
    "MistreatmentAnalyst": "誤治傳變師",
}


class Council:
    def __init__(self, client: Optional[LLMClient] = None,
                 registry: Optional[ToolRegistry] = None,
                 llm_specialists: bool = True):
        self.client = client or get_client()
        self.registry = registry or get_registry()
        # when a real model is available, each specialist adds a short
        # grounded comment on its own tool evidence (citation-checked)
        self.llm_specialists = llm_specialists

    def _specialist_comment(self, msg: CouncilMessage) -> None:
        """Append an LLM remark grounded in this specialist's tool data."""
        if not (self.llm_specialists and self.client.available):
            return
        try:
            data = json.dumps(msg.data, ensure_ascii=False)[:2500]
            remark = self.client.complete(
                "你是《傷寒論》合議庭的" + msg.role_cn +
                "。基於下方工具證據，用一至三句話給出你的專業判斷。"
                "只可使用證據中的事實；引用條文須附 clause_id"
                "（僅可取自證據）；證據不足就明說。",
                f"問題相關證據（JSON）：\n{data}", task="synthesize").strip()
            if not remark:
                return
            guard = CitationGuard(self.registry.art.clause_store())
            report = guard.check(remark)
            if report.unsupported_ids:
                remark += "（⚠️ 含未核實條文編號，請以文末核驗為準）"
            msg.content += "\n💬 " + remark
            msg.evidence_ids = list(dict.fromkeys(
                msg.evidence_ids + report.verified_ids))
            msg.data["llm_remark"] = remark
        except Exception:
            pass  # specialist prose is optional; never break the council

    # ------------------------------------------------------------------
    def _infer_role(self, question: str, role: Optional[str]) -> str:
        if role in safety.ROLES:
            return role
        from ..rag.skill_rag import SkillRAG
        try:
            return SkillRAG().infer_role(question, role)
        except Exception:
            return "doctor"

    def deliberate(self, question: str, role: Optional[str] = None) -> Dict[str, Any]:
        role = self._infer_role(question, role)
        messages: List[CouncilMessage] = []
        evidence_ids: List[str] = []

        # patient guard up front
        if role == "patient":
            guard = safety.patient_intent_guard(question)
            if guard:
                messages.append(CouncilMessage(
                    "Critic", "安全治理官", "critique",
                    content="患者語境涉及診斷/處方/劑量，已攔截。", data=guard))
                out = safety.governed(guard, "patient")
                out.update({"question": question, "backend": self.client.backend,
                            "council": [m.to_dict() for m in messages],
                            "evidence_clause_ids": []})
                return out

        q = normalize_query(question)

        # 1 — Planner -----------------------------------------------------
        plan = self._plan(q, question)
        messages.append(CouncilMessage(
            "Planner", "調度規劃師", "plan",
            content="擬調度：" + "、".join(SPECIALISTS[s] for s in plan["specialists"]) +
                    f"（識別：症狀{len(plan['symptoms'])}、脈{len(plan['pulse'])}、"
                    f"方{len(plan['formulas'])}、經{plan['channel'] or '—'}）",
            data=plan))

        # 2 — Retriever ---------------------------------------------------
        retr = self.registry.call("shanghan_search", {"query": question, "top_k": 6,
                                                       "expand": True})
        hits = retr.get("hits", [])
        ev = [h["clause_id"] for h in hits]
        evidence_ids += ev
        messages.append(CouncilMessage(
            "Retriever", "原文取證師", "retrieve",
            content=f"檢索到 {len(hits)} 條相關條文（A 原文直述）。",
            evidence_ids=ev, tool_calls=[{"tool": "shanghan_search"}],
            data={"hits": hits}))

        # 3 — Specialists -------------------------------------------------
        specialist_findings: List[Dict] = []
        for spec in plan["specialists"]:
            msg = self._run_specialist(spec, plan, role)
            if msg:
                self._specialist_comment(msg)
                messages.append(msg)
                evidence_ids += msg.evidence_ids
                specialist_findings.append({"agent": spec, "summary": msg.content,
                                            "data": msg.data})

        # 4 — Critic ------------------------------------------------------
        critic_msg, contraindication_notes = self._critique(specialist_findings, evidence_ids)
        messages.append(critic_msg)

        # 5 — Synthesizer -------------------------------------------------
        evidence_ids = list(dict.fromkeys(evidence_ids))
        final = self._synthesize(question, role, plan, specialist_findings,
                                 contraindication_notes, hits)
        guard = CitationGuard(self.registry.art.clause_store())
        report = guard.check(final)
        final = guard.annotate(final, report)
        messages.append(CouncilMessage(
            "Synthesizer", "合議綜合官", "synthesize",
            content="已綜合各專家意見並核驗引用。",
            evidence_ids=report.verified_ids,
            data={"citation_report": report.to_dict()}))

        payload = {
            "question": question, "backend": self.client.backend,
            "answer": final, "role": role,
            "council": [m.to_dict() for m in messages],
            "evidence_clause_ids": report.verified_ids,
            "citation_report": report.to_dict(),
            "specialists": plan["specialists"],
        }
        return safety.governed(payload, role)

    # ------------------------------------------------------------------
    def _plan(self, q: str, raw: str) -> Dict:
        from ..extract.entities import EntityExtractor
        res = EntityExtractor().extract(q)
        formulas = [n for n in sorted(lexicon.FORMULA_SEEDS, key=len, reverse=True)
                    if n in q][:3]
        channel = next((c for c in lexicon.CHANNEL_IN_TEXT.values() if c in q
                        or c[:-1] in q), None)
        specialists: List[str] = []
        if res.symptoms or res.pulse:
            specialists.append("FormulaAnalyst")
        if len(formulas) >= 2 or any(k in q for k in ("鑒別", "區別", "不同", "對比")):
            specialists.append("DifferentialAnalyst")
        if channel or any(k in q for k in ("六經", "提綱", "綱領", "內部結構")):
            specialists.append("ChannelAnalyst")
        if any(k in q for k in ("誤治", "誤下", "誤汗", "誤吐", "火逆", "壞病", "變證", "傳變")):
            specialists.append("MistreatmentAnalyst")
        if not specialists:
            specialists.append("FormulaAnalyst")
        return {"symptoms": res.symptoms, "pulse": res.pulse, "formulas": formulas,
                "channel": channel, "specialists": specialists,
                "mistreatment": res.mistreatment_types}

    def _run_specialist(self, spec: str, plan: Dict, role: str) -> Optional[CouncilMessage]:
        cn = SPECIALISTS[spec]
        if spec == "FormulaAnalyst":
            if not (plan["symptoms"] or plan["pulse"]):
                return None
            out = self.registry.call("shanghan_match_formula",
                                     {"symptoms": plan["symptoms"], "pulse": plan["pulse"],
                                      "top_k": 4})
            matches = out.get("matched_formula_patterns", [])
            ev = [e["clause_id"] for m in matches for e in m.get("evidence", [])]
            top = "、".join(f"{m['formula']}({m['match_score']})" for m in matches[:3])
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"候選方證：{top or '無顯著匹配'}。",
                                  evidence_ids=ev,
                                  tool_calls=[{"tool": "shanghan_match_formula"}],
                                  data={"matches": matches})
        if spec == "DifferentialAnalyst":
            formulas = plan["formulas"]
            if len(formulas) < 2:
                # derive from formula analyst if available later; try top matches
                fm = self.registry.call("shanghan_match_formula",
                                        {"symptoms": plan["symptoms"], "pulse": plan["pulse"],
                                         "top_k": 2})
                formulas = [m["formula"] for m in fm.get("matched_formula_patterns", [])][:2]
            if len(formulas) < 2:
                return CouncilMessage(spec, cn, "analyze",
                                      content="可鑒別方不足兩個，略過鑒別。")
            out = self.registry.call("shanghan_differential", {"formulas": formulas})
            d = out.get("differential", {})
            ev = d.get("supporting_clauses", [])
            disc = "；".join(d.get("key_discriminators", [])[:3])
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"鑒別 {' vs '.join(formulas)}：{disc}",
                                  evidence_ids=ev,
                                  tool_calls=[{"tool": "shanghan_differential"}],
                                  data={"differential": d})
        if spec == "ChannelAnalyst":
            channel = plan["channel"] or "太陽病"
            out = self.registry.call("shanghan_six_channel", {"channel": channel})
            if out.get("error"):
                return CouncilMessage(spec, cn, "analyze", content=out["error"])
            ev = [out.get("outline_clause_id", "")]
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"{channel}：{out.get('summary','')[:60]}…",
                                  evidence_ids=[e for e in ev if e],
                                  tool_calls=[{"tool": "shanghan_six_channel"}],
                                  data={"six_channel": out})
        if spec == "MistreatmentAnalyst":
            out = self.registry.call("shanghan_mistreatment", {"query": plan.get("channel") or ""})
            paths = out.get("paths", [])
            ev = [c for p in paths for c in p.get("clauses", [])]
            sample = "；".join(f"{p['mistreatment']}→{p['resulting_pattern']}→"
                               f"{'、'.join(p['rescue_formulas'][:1])}" for p in paths[:3])
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"誤治路徑：{sample}",
                                  evidence_ids=ev[:6],
                                  tool_calls=[{"tool": "shanghan_mistreatment"}],
                                  data={"paths": paths})
        return None

    def _critique(self, findings: List[Dict], evidence_ids: List[str]):
        notes: List[str] = []
        for f in findings:
            for m in f.get("data", {}).get("matches", []) or []:
                if m.get("contraindications"):
                    c = m["contraindications"][0]
                    notes.append(f"{m['formula']} 有禁忌：{c.get('condition','')[:24]}…"
                                 f"（{c.get('clause_id','')}）")
                if m.get("conflicts"):
                    notes.append(f"{m['formula']}：{m['conflicts'][0]}")
        verified = len(set(evidence_ids))
        content = f"已歸集證據 {verified} 條，逐一回源；"
        content += ("發現需提示的禁忌/衝突：" + "；".join(notes[:3])) if notes else "未見明顯禁忌衝突。"
        return CouncilMessage("Critic", "安全治理官", "critique", content=content,
                              data={"contraindication_notes": notes}), notes

    def _synthesize(self, question, role, plan, findings, contraindication_notes,
                    hits) -> str:
        # gather evidence for the synthesizer
        evidence: List[Dict] = list(hits)
        for f in findings:
            for m in f.get("data", {}).get("matches", []) or []:
                evidence.extend(m.get("evidence", []))
        # LLM prose if available, else deterministic template
        if self.client.available:
            try:
                summary = "\n".join(f"[{f['agent']}] {f['summary']}" for f in findings)
                prose = self.client.synthesize(
                    question + "\n專家findings:\n" + summary, evidence, role)
                if contraindication_notes:
                    prose += "\n\n⚠️ 禁忌提示：" + "；".join(contraindication_notes[:3])
                return prose
            except Exception:
                pass
        # deterministic synthesis
        lines = [f"（多智能體合議 · {self.client.backend} 後端 · 角色：{role}）", ""]
        for f in findings:
            lines.append(f"▸ {SPECIALISTS.get(f['agent'], f['agent'])}：{f['summary']}")
        if contraindication_notes:
            lines.append("")
            lines.append("⚠️ 禁忌/衝突提示：" + "；".join(contraindication_notes[:3]))
        # representative evidence
        ev_ids = list(dict.fromkeys(e.get("clause_id") for e in evidence if e.get("clause_id")))[:5]
        if ev_ids:
            lines.append("")
            lines.append("證據條文：" + "、".join(ev_ids))
        if role == "doctor":
            lines.append("")
            lines.append("（以上為古籍方證輔助合議，不替代醫師臨床判斷。）")
        return "\n".join(lines)
