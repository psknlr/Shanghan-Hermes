"""EvidenceBinder — 把回答拆成 claims，逐句綁定到本輪工具證據.

CitationGuard 回答「引用的編號是否真實、是否來自本輪取證」；EvidenceBinder
再進一步回答「這句話本身有沒有證據支撐、支撐到什麼程度、屬於哪個證據層」：

  claim            回答中的一個結論句
  evidence         支撐該句的 clause_id（僅限本輪工具結果中出現過的）
  support_type     direct（句中引用且術語落在條文內）/ cited_low_overlap
                   （引用了但句子內容與條文重合度低）/ inferred（未引用，
                   由詞彙對齊推定）/ ungrounded（找不到證據）
  evidence_layer   A 原文直述／B 版本異文／C 注家解釋／D 歸納／D/E 病機推理
                   （句中出現後世病機術語一律降為 D/E，不得冒充原文）
  confidence       上述判定的啟發式置信度

聚合指標 claim_grounding_rate 是「無證據鏈，不成回答」的句級量化。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .. import lexicon
from ..textutil import normalize_query, similarity
from .citation_guard import RE_CLAUSE_ID

RE_SENT = re.compile(r"[。；;\n]+")
# lines that are metadata/boilerplate, not claims
RE_SKIP = re.compile(
    r"^(（|【|⚠️|—|-{2,}|證據條文|支持條文|已核實條文|證據：|評測基準|"
    r"\s*$)|(僅供|不替代|請遵醫囑|輔助性質)")
RE_COMMENTARY = re.compile(r"注家|注本|成無己|柯琴|尤怡|方有執|錢潢|黃元御|詮釋")
RE_VARIANT = re.compile(r"異文|桂林古本|桂本|千金翼|版本差")

# domain terms used for lexical claim↔clause alignment
_TERMS: List[str] = sorted(
    set(lexicon.SYMPTOMS) | set(lexicon.PULSE_NAMED_PATTERNS)
    | set(lexicon.DISEASE_PATTERNS) | set(lexicon.FORMULA_SEEDS),
    key=lambda t: (-len(t), t))
_POSTHOC = sorted(lexicon.POSTHOC_TERMS, key=lambda t: (-len(t), t))


class EvidenceBinder:
    def __init__(self, clause_store: Dict):
        self.store = clause_store

    # ------------------------------------------------------------------
    def bind(self, answer: str, tool_results: List[Dict]) -> Dict:
        """Split *answer* into claims and align each to this round's
        evidence. ``tool_results`` is the agent's [{tool, arguments,
        result}] list; only clause_ids appearing there count as evidence."""
        import json as _json
        blob = _json.dumps([t.get("result", {}) for t in (tool_results or [])],
                           ensure_ascii=False)
        round_ids = list(dict.fromkeys(RE_CLAUSE_ID.findall(blob)))
        round_texts = {cid: normalize_query(self.store[cid].clean_text)
                       for cid in round_ids if cid in self.store}

        claims: List[Dict] = []
        for raw in RE_SENT.split(answer or ""):
            sent = raw.strip().lstrip("-▸·• ").strip()
            if len(sent) < 8 or RE_SKIP.search(sent):
                continue
            claims.append(self._bind_one(sent, round_texts))
        n = len(claims)
        grounded = sum(1 for c in claims if c["support_type"] != "ungrounded")
        return {
            "claims": claims,
            "n_claims": n,
            "n_grounded": grounded,
            "claim_grounding_rate": round(grounded / n, 4) if n else 0.0,
            "ungrounded_claims": [c["claim"] for c in claims
                                  if c["support_type"] == "ungrounded"][:5],
            "round_evidence_ids": round_ids,
            # 誠實邊界（九輪 P0-5）：本綁定器的 verifier 是詞彙重合，
            # 只能證明「句子與條文共享術語」，**不能證明語義蘊含**——
            # 引用合法 ≠ 結論成立。claim_grounding_rate 是詞彙級下界指標；
            # supports/contradicts 級 entailment 需模型後端（見路線圖）。
            "verifier": "lexical_overlap_v1",
            "verifier_note": "詞彙重合校驗；非語義蘊含。逐句 relation 見 "
                             "claims[].relation（supports_lexical/mentions/"
                             "none），均為詞彙級判定。",
        }

    # ------------------------------------------------------------------
    def _bind_one(self, sent: str, round_texts: Dict[str, str]) -> Dict:
        norm = normalize_query(sent)
        cited = [cid for cid in dict.fromkeys(RE_CLAUSE_ID.findall(sent))
                 if cid in round_texts]
        terms = [t for t in _TERMS if t in norm][:8]

        def overlap(cid: str) -> float:
            text = round_texts[cid]
            if terms:
                return sum(1 for t in terms if t in text) / len(terms)
            return similarity(norm, text)

        evidence, support = list(cited), "ungrounded"
        best_score = max((overlap(c) for c in cited), default=0.0)
        if cited:
            support = "direct" if best_score >= 0.3 else "cited_low_overlap"
        elif round_texts:
            # no explicit citation: lexical alignment against round evidence
            ranked = sorted(round_texts, key=overlap, reverse=True)
            if ranked and overlap(ranked[0]) >= 0.5 and terms:
                evidence, support = [ranked[0]], "inferred"
                best_score = overlap(ranked[0])

        posthoc = [t for t in _POSTHOC if t in norm]
        if posthoc:
            layer = "D/E"       # 病機歸納，不得冒充原文
        elif RE_COMMENTARY.search(norm):
            layer = "C"
        elif RE_VARIANT.search(norm):
            layer = "B"
        elif support == "direct":
            layer = "A"
        else:
            layer = "D"
        confidence = {"direct": 0.9, "cited_low_overlap": 0.6,
                      "inferred": 0.5, "ungrounded": 0.2}[support]
        if posthoc:
            confidence = min(confidence, 0.65)
        # 結構化 Claim–Evidence 記錄（九輪 P0-5 第一步）：關係與核驗狀態
        # 顯式化且**如實標注核驗手段**——lexical 不冒充 entailment
        import hashlib
        relation = ("supports_lexical" if support == "direct"
                    else "mentions" if support in ("cited_low_overlap",
                                                   "inferred")
                    else "none")
        verification = ("verified_lexical" if support == "direct"
                        else "weak_lexical" if relation == "mentions"
                        else "unverified")
        return {"claim": sent[:120],
                "claim_id": hashlib.sha256(sent.encode()).hexdigest()[:10],
                "evidence": evidence,
                "evidence_links": [{"evidence_id": cid, "relation": relation,
                                    "entailment_score": round(best_score, 3),
                                    "verifier": "lexical_overlap_v1",
                                    "verification_status": verification}
                                   for cid in evidence],
                "support_type": support, "evidence_layer": layer,
                "posthoc_terms": posthoc[:3],
                "confidence": round(confidence, 2)}
