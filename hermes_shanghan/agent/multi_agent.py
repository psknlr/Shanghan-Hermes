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
            # the remark may only cite THIS specialist's own tool evidence
            from .citation_guard import RE_CLAUSE_ID
            allowed = list(dict.fromkeys(RE_CLAUSE_ID.findall(data)))
            report = guard.check(remark, allowed_ids=allowed or None)
            if report.unsupported_ids or report.outside_evidence_ids:
                remark += "（⚠️ 含未核實或超出本專家證據的條文編號，請以文末核驗為準）"
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

        # patient guard up front: red-flag triage, then intent guard
        if role == "patient":
            triage = safety.red_flag_triage(question)
            if triage:
                messages.append(CouncilMessage(
                    "SafetyTriage", "安全分診官", "critique",
                    content="患者語境出現紅旗信號，優先就醫提示。", data=triage))
                out = safety.governed(triage, "patient")
                out.update({"question": question, "backend": self.client.backend,
                            "council": [m.to_dict() for m in messages],
                            "evidence_clause_ids": []})
                return out
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

        # hard role isolation: patient deliberations see only safe tools
        reg = self.registry.for_role(role)

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
        retr = reg.call("shanghan_search", {"query": question, "top_k": 6,
                                                       "expand": True})
        hits = retr.get("hits", [])
        ev = [h["clause_id"] for h in hits]
        evidence_ids += ev
        messages.append(CouncilMessage(
            "Retriever", "原文取證師", "retrieve",
            content=f"檢索到 {len(hits)} 條相關條文（A 原文直述）。",
            evidence_ids=ev, tool_calls=[{"tool": "shanghan_search"}],
            data={"hits": hits}))

        # 3 — Specialists: each emits an INDEPENDENT structured judgment ----
        specialist_findings: List[Dict] = []
        judgments: List[Dict] = []
        must_verify: List[str] = []
        for spec in plan["specialists"]:
            msg = self._run_specialist(spec, plan, role, reg)
            if msg:
                self._specialist_comment(msg)
                messages.append(msg)
                evidence_ids += msg.evidence_ids
                specialist_findings.append({"agent": spec, "summary": msg.content,
                                            "data": msg.data})
                if msg.data.get("judgment"):
                    judgments.append(msg.data["judgment"])
                must_verify += msg.data.get("must_verify", [])

        # 4 — Critic ------------------------------------------------------
        critic_msg, contraindication_notes = self._critique(specialist_findings, evidence_ids)
        messages.append(critic_msg)

        # 4b — ConsensusJudge: 共識/分歧/需補充 + 評分裁決 ------------------
        from .consensus import ConsensusJudge, render_adjudication
        adjudication = ConsensusJudge().adjudicate(
            judgments, contraindication_notes, must_verify)
        messages.append(CouncilMessage(
            "ConsensusJudge", "合議裁決官", "adjudicate",
            content=(f"主導假設：{adjudication['dominant_hypothesis'] or '—'}；"
                     f"置信 {adjudication['final_confidence']}"
                     f"（{adjudication['decision']}）"),
            data=adjudication))

        # 5 — Synthesizer -------------------------------------------------
        evidence_ids = list(dict.fromkeys(evidence_ids))
        final = self._synthesize(question, role, plan, specialist_findings,
                                 contraindication_notes, hits)
        final = final.rstrip() + "\n\n" + render_adjudication(adjudication)
        guard = CitationGuard(self.registry.art.clause_store())
        # 嚴格接地：合議答案只可引用本輪各專家取回的證據
        report = guard.check(final,
                             allowed_ids=evidence_ids if evidence_ids else None)
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
            "judgments": judgments,
            "consensus": adjudication,
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

    def _run_specialist(self, spec: str, plan: Dict, role: str,
                        reg=None) -> Optional[CouncilMessage]:
        reg = reg or self.registry
        cn = SPECIALISTS[spec]
        if spec == "FormulaAnalyst":
            if not (plan["symptoms"] or plan["pulse"]):
                return None
            out = reg.call("shanghan_match_formula",
                                     {"symptoms": plan["symptoms"], "pulse": plan["pulse"],
                                      "top_k": 4})
            matches = out.get("matched_formula_patterns", [])
            ev = [e["clause_id"] for m in matches for e in m.get("evidence", [])]
            top = "、".join(f"{m['formula']}({m['match_score']})" for m in matches[:3])
            # independent structured judgment（多假設 + 追問由 HypothesisManager 供給）
            judgment, must_verify, hyps = None, [], []
            try:
                if role == "patient":
                    raise RuntimeError("hypotheses disabled for patient role")
                from .hypothesis import HypothesisManager
                hyp = HypothesisManager(self.registry).analyze(
                    plan["symptoms"], plan["pulse"])
                hyps = hyp.get("hypotheses", [])
                must_verify = hyp.get("clarifying_questions", [])[:3]
            except Exception:
                hyp = {}
            if matches:
                m0 = matches[0]
                close = [h["formula"] for h in hyps[1:]
                         if m0.get("match_score", 0) - h.get("score", 0) < 0.15]
                judgment = {
                    "agent": spec,
                    "hypothesis": f"{m0['formula']}證可能性較高",
                    "support": [x.split("：", 1)[-1]
                                for x in m0.get("matched_findings", [])][:5],
                    "against": m0.get("conflicts", []),
                    "evidence": [e["clause_id"] for e in m0.get("evidence", [])],
                    "confidence": m0.get("match_score", 0.0),
                    "data_channel_scope": m0.get("six_channel", ""),
                    "close_alternatives": close[:2],
                }
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"候選方證：{top or '無顯著匹配'}。",
                                  evidence_ids=ev,
                                  tool_calls=[{"tool": "shanghan_match_formula"}],
                                  data={"matches": matches, "judgment": judgment,
                                        "hypotheses": hyps,
                                        "must_verify": must_verify})
        if spec == "DifferentialAnalyst":
            formulas = plan["formulas"]
            if len(formulas) < 2:
                # derive from formula analyst if available later; try top matches
                fm = reg.call("shanghan_match_formula",
                                        {"symptoms": plan["symptoms"], "pulse": plan["pulse"],
                                         "top_k": 2})
                formulas = [m["formula"] for m in fm.get("matched_formula_patterns", [])][:2]
            if len(formulas) < 2:
                return CouncilMessage(spec, cn, "analyze",
                                      content="可鑒別方不足兩個，略過鑒別。")
            out = reg.call("shanghan_differential", {"formulas": formulas})
            d = out.get("differential", {})
            ev = d.get("supporting_clauses", [])
            disc = "；".join(d.get("key_discriminators", [])[:3])
            judgment = {
                "agent": spec,
                "hypothesis": f"需鑒別 {' 與 '.join(formulas)}",
                "support": d.get("key_discriminators", [])[:3],
                "against": [],
                "evidence": ev[:6],
                "confidence": {"gold": 0.85, "silver": 0.7,
                               "bronze": 0.55}.get(d.get("release_level"), 0.6),
            }
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"鑒別 {' vs '.join(formulas)}：{disc}",
                                  evidence_ids=ev,
                                  tool_calls=[{"tool": "shanghan_differential"}],
                                  data={"differential": d, "judgment": judgment})
        if spec == "ChannelAnalyst":
            channel = plan["channel"] or "太陽病"
            out = reg.call("shanghan_six_channel", {"channel": channel})
            if out.get("error"):
                return CouncilMessage(spec, cn, "analyze", content=out["error"])
            ev = [out.get("outline_clause_id", "")]
            judgment = {
                "agent": spec,
                "hypothesis": f"{channel}方向",
                "support": [f"提綱：{out.get('outline_text', '')[:30]}"],
                "against": [],
                "evidence": [e for e in ev if e],
                # explicit channel mention in the question beats a default
                "confidence": 0.75 if plan["channel"] else 0.5,
                "data_channel": channel,
            }
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"{channel}：{out.get('summary','')[:60]}…",
                                  evidence_ids=[e for e in ev if e],
                                  tool_calls=[{"tool": "shanghan_six_channel"}],
                                  data={"six_channel": out, "judgment": judgment})
        if spec == "MistreatmentAnalyst":
            out = reg.call("shanghan_mistreatment", {"query": plan.get("channel") or ""})
            paths = out.get("paths", [])
            ev = [c for p in paths for c in p.get("clauses", [])]
            sample = "；".join(f"{p['mistreatment']}→{p['resulting_pattern']}→"
                               f"{'、'.join(p['rescue_formulas'][:1])}" for p in paths[:3])
            judgment = {
                "agent": spec,
                "hypothesis": "存在誤治傳變風險路徑" if paths else "",
                "support": [sample] if sample else [],
                "against": [],
                "evidence": ev[:6],
                "confidence": 0.6 if paths else 0.2,
            } if paths else None
            return CouncilMessage(spec, cn, "analyze",
                                  content=f"誤治路徑：{sample}",
                                  evidence_ids=ev[:6],
                                  tool_calls=[{"tool": "shanghan_mistreatment"}],
                                  data={"paths": paths, "judgment": judgment})
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
