"""ServiceContext — the framework-agnostic API surface behind the web console.

Every method returns a JSON-serializable dict and reuses the existing engine
(RAG, apps, agent, council, paper, research). Artifacts are lazy-loaded once
and shared; the HTTP layer is a thin adapter over this.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import config
from ..schemas import read_jsonl


class ServiceContext:
    def __init__(self):
        self._art = None
        self._clause_rag = None
        self._skill_rag = None
        self._matcher = None
        self._registry = None
        self._llm = None

    # -- lazy resources -------------------------------------------------
    @property
    def art(self):
        if self._art is None:
            from ..orchestrator import Artifacts
            self._art = Artifacts()
        return self._art

    @property
    def clause_rag(self):
        if self._clause_rag is None:
            from ..rag.clause_rag import ClauseRAG
            self._clause_rag = ClauseRAG.load()
        return self._clause_rag

    @property
    def matcher(self):
        if self._matcher is None:
            from ..apps.doctor import FormulaMatcher
            self._matcher = FormulaMatcher(self.art.formula_rules, self.art.clause_store())
        return self._matcher

    @property
    def registry(self):
        if self._registry is None:
            from ..agent.tools import get_registry
            self._registry = get_registry()
        return self._registry

    @property
    def llm(self):
        if self._llm is None:
            from ..llm.client import get_client
            self._llm = get_client()
        return self._llm

    def warm(self):
        _ = self.clause_rag, self.art.formula_rules, self.registry

    @staticmethod
    def ready() -> bool:
        return (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists()

    # -- dashboard ------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        rules = read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
        clauses = read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")
        levels = Counter(r["autonomous_review"]["release_level"] for r in rules)
        types = Counter(r["rule_type"] for r in rules)
        return {
            "clauses": len(clauses),
            "canonical": sum(1 for c in clauses if c["text_type"] == "original_clause"),
            "initial_rules": len(rules),
            "release_levels": dict(levels),
            "rule_types": dict(types.most_common()),
            "formula_pattern_rules": len(self.art.formula_rules),
            "six_channel_rules": len(self.art.six_channel_rules),
            "therapy_rules": len(read_jsonl(config.RULES_THERAPY_DIR / "therapy_rules.jsonl")),
            "mistreatment_rules": len(self.art.mistreatment_rules),
            "differential_rules": len(self.art.differential_rules),
            "merged_rules": len(self.art.merged_rules),
            "variant_rules": len(self.art.variant_rules),
            "commentary_rules": len(self.art.commentary_rules),
            "clause_relations": len(read_jsonl(config.RELATION_DIR / "clause_relations.jsonl")),
            "audits": len(read_jsonl(config.AUDIT_DIR / "audit_log.jsonl")),
            "skills": self._skill_count(),
        }

    def _skill_count(self) -> int:
        import json
        m = config.SKILLS_DIR / "skills_manifest.json"
        if m.exists():
            try:
                return json.loads(m.read_text(encoding="utf-8")).get("total_dirs", 0)
            except Exception:
                return 0
        return 0

    def llm_status(self) -> Dict[str, Any]:
        from ..llm.config import RECOMMENDED_MODELS
        st = self.llm.status()
        st["recommended_models"] = RECOMMENDED_MODELS
        return st

    # -- retrieval / clause --------------------------------------------
    def search(self, query: str, top_k: int = 8, six_channel: str = None,
               formula: str = None, field: str = None, expand: bool = False) -> Dict:
        hits = self.clause_rag.search(query, top_k=top_k, six_channel=six_channel or None,
                                      formula=formula or None, field=field or None,
                                      expand_relations=expand)
        return {"query": query, "hits": hits, "count": len(hits)}

    def explain_clause(self, ref, role: str = "student") -> Dict:
        from .. import safety
        c = self.clause_rag.get_clause(ref)
        if c is None:
            return {"error": f"未找到條文 {ref}"}
        rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
                 if r["clause_id"] == c.clause_id]
        variants = [v for v in self.art.variant_rules if v.clause_id == c.clause_id]
        comments = [v for v in self.art.commentary_rules if v.clause_id == c.clause_id]
        payload = {
            "clause_id": c.clause_id, "clause_number": c.clause_number,
            "chapter": c.chapter, "six_channel": c.six_channel,
            "layer_label": config.LAYER_LABEL.get(c.layer, ""),
            "text": c.clean_text,
            "entities": {"symptoms": c.symptoms, "negated_findings": c.negated_findings,
                         "pulse": c.pulse, "formulas": c.formula_names,
                         "disease_patterns": c.disease_patterns,
                         "therapy": c.therapy_terms,
                         "contraindications": c.contraindication_terms,
                         "mistreatment": c.mistreatment_terms,
                         "prognosis": c.prognosis_terms},
            "formula_blocks": [fb.to_dict() for fb in c.formula_blocks],
            "initial_rules": [{"id": r["initial_rule_id"], "type": r["rule_type"],
                               "strength": r.get("prescription_strength", ""),
                               "release": r["autonomous_review"]["release_level"],
                               "interpretation": r.get("interpretation", "")}
                              for r in rules],
            "relations": self.clause_rag.related(c.clause_id, limit=10),
            "variants": [{"book": v.variant_book, "text": v.variant_text,
                          "similarity": v.similarity,
                          "differences": v.notable_differences} for v in variants],
            "commentaries": [{"commentator": v.commentator,
                              "text": v.commentary_text[:400]} for v in comments],
        }
        return safety.governed(payload, role)

    # -- apps -----------------------------------------------------------
    def match(self, symptoms: List[str], pulse: List[str] = None,
              six_channel: str = None, top_k: int = 5) -> Dict:
        return self.matcher.match(symptoms=symptoms, pulse=pulse or [],
                                  six_channel=six_channel or None, top_k=top_k)

    def differential(self, formulas: List[str]) -> Dict:
        from .. import safety
        from ..textutil import normalize_query
        names = [normalize_query(f) for f in formulas]
        cands = [d for d in self.art.differential_rules if set(names) <= set(d.formulas)]
        if not cands:
            cands = [d for d in self.art.differential_rules
                     if len(set(names) & set(d.formulas)) >= 2]
        if not cands:
            from ..induce.differential import DifferentialInducer
            one = DifferentialInducer(self.art.formula_rules)._build_one(names, 999)
            cands = [one] if one else []
        if not cands:
            return {"error": "無法構建該鑒別對"}
        return safety.governed({"differential": cands[0].to_dict()}, "doctor")

    def teach(self, channel: str) -> Dict:
        from ..apps.teaching import TeachingBuilder
        tb = TeachingBuilder(self.art.clauses, self.art.six_channel_rules,
                             self.art.formula_rules, self.art.mistreatment_rules)
        return tb.lesson(channel)

    def mistreatment(self, query: str = None) -> Dict:
        return self.registry.call("shanghan_mistreatment", {"query": query or ""})

    def patient(self, question: str) -> Dict:
        from ..apps.patient import PatientEducator
        edu = PatientEducator(self.art.six_channel_rules, self.art.clause_store())
        return edu.explain(question)

    def formula_rule(self, formula: str) -> Dict:
        return self.registry.call("shanghan_formula_rule", {"formula": formula})

    def list_formulas(self) -> Dict:
        return {"formulas": sorted(r.formula for r in self.art.formula_rules)}

    def channels(self) -> Dict:
        return {"channels": [r.six_channel for r in self.art.six_channel_rules]}

    def skills(self) -> Dict:
        from ..rag.skill_rag import SkillRAG
        return {"skills": SkillRAG().describe()}

    # -- research / paper ----------------------------------------------
    def research(self, topic: str, outputs: List[str] = None) -> Dict:
        from ..apps.research import ResearchMiner
        miner = ResearchMiner(self.art.clauses, self.art.formula_rules,
                              self.art.mistreatment_rules)
        return miner.run_topic(topic, outputs=outputs or ["rules", "network", "paper_outline"])

    def paper(self, paper_type: str = "formula_pattern", topic: str = "",
              use_llm: bool = True) -> Dict:
        from ..paper.writer import PaperWriter
        writer = PaperWriter(self.art.clauses, self.art.initial_rules,
                             self.art.formula_rules, self.art.six_channel_rules,
                             self.art.mistreatment_rules, self.art.differential_rules,
                             commentary_rules=self.art.commentary_rules,
                             llm_client=self.llm)
        path = writer.generate(paper_type=paper_type, topic=topic or "",
                               use_llm=use_llm)
        meta = {}
        meta_path = path.parent / "paper_meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"manuscript_path": str(path),
                "manuscript": path.read_text(encoding="utf-8"),
                "meta": meta}

    def complex(self, question: str, role: str = None) -> Dict:
        from ..agent.complex_agent import ComplexAgent
        return ComplexAgent(client=self.llm,
                            registry=self.registry).solve(question, role=role)

    def chat(self, question: str, session_id: str = "default",
             role: str = None) -> Dict:
        from ..agent.session import AgentSession
        if not hasattr(self, "_sessions"):
            self._sessions = {}
        sess = self._sessions.setdefault(
            session_id, AgentSession(client=self.llm, registry=self.registry))
        return sess.ask(question, role=role)

    def deep_research(self, topic: str, rounds: int = 3) -> Dict:
        from ..agent.research_loop import DeepResearcher
        from .. import safety
        d = DeepResearcher(client=self.llm, registry=self.registry,
                           max_rounds=rounds).run(topic)
        return safety.governed(d, "researcher")

    # -- agent / council -----------------------------------------------
    def agent(self, question: str, role: str = None, max_steps: int = 5) -> Dict:
        from ..agent.agent import ShanghanAgent
        return ShanghanAgent(client=self.llm, registry=self.registry,
                             max_steps=max_steps).ask(question, role=role)

    def council(self, question: str, role: str = None) -> Dict:
        from ..agent.multi_agent import Council
        return Council(client=self.llm, registry=self.registry).deliberate(question, role=role)

    def tool_call(self, name: str, arguments: Dict) -> Dict:
        return self.registry.call(name, arguments or {})

    def trace(self, query_type: str, ref: str) -> Dict:
        from ..trace.chains import trace_dispatch
        return trace_dispatch(query_type, ref)

    def tools(self) -> Dict:
        from ..integrations.tool_specs import openai_tool_specs
        return {"tools": openai_tool_specs()}

    def gold_sample(self, n: int = 20, stratify: bool = True) -> Dict:
        from ..trace.goldset import build_sample
        return build_sample(n=n, stratify=stratify)   # 不落盤，rows 隨響應返回

    def gold_eval(self, rows) -> Dict:
        from ..trace.goldset import evaluate_rows
        return evaluate_rows(rows or [])

    def herb(self, name: str) -> Dict:
        return self.registry.call("shanghan_herb_profile", {"herb": name})

    def formula_explain(self, name: str) -> Dict:
        return self.registry.call("shanghan_formula_explain", {"formula": name})


_SERVICE: Optional[ServiceContext] = None


def get_service() -> ServiceContext:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ServiceContext()
    return _SERVICE
