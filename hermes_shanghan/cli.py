"""Hermes-Shanghanlun CLI.

Commands:
  pipeline          run all workflows end to end
  ingest            corpus manifest only
  stats             show pipeline statistics
  search QUERY      Classical Text RAG over clauses
  clause N          show clause N with entities, rules, relations
  explain-clause N  full clause explanation (原文/異文/注/規則/關係)
  match             doctor-mode formula matching
  ask QUESTION      Skill RAG question answering (role-aware)
  teach CHANNEL     six-channel lesson + quiz
  differential F1 F2…  formula contrast table
  research TOPIC    research mining outputs
  paper             generate a manuscript
  skills            list compiled skills
  trace REF         deep provenance chains (clause/formula/claim/school/text)
  trace-network     citation-network scientometrics (cocitation/slices/bursts)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import config, safety
from .schemas import read_jsonl


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def _need_pipeline():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        print("規則庫未生成，請先運行: hermes-shanghan pipeline", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
def cmd_pipeline(args):
    from .orchestrator import run_pipeline
    stats = run_pipeline(verbose=not args.quiet,
                         use_llm_extractor=getattr(args, "llm_extract", False),
                         use_llm_critic=getattr(args, "llm_critic", False))
    _print(stats)


def cmd_llm_status(args):
    from .llm.client import get_client
    from .llm.config import RECOMMENDED_MODELS
    client = get_client()
    st = client.status()
    st["recommended_models"] = RECOMMENDED_MODELS
    st["how_to_enable"] = ("pip install litellm 並設置 ANTHROPIC_API_KEY（或 OPENAI_API_KEY 等），"
                           "可選 HERMES_LLM_MODEL 指定模型；無配置時自動使用 local 確定性後端。")
    _print(st)


def cmd_agent(args):
    _need_pipeline()
    from .agent.agent import ShanghanAgent
    agent = ShanghanAgent(max_steps=args.max_steps)
    out = agent.ask(args.question, role=args.role)
    if args.answer_only:
        print(out.get("answer", ""))
    else:
        _print(out)


def cmd_llm_extract(args):
    _need_pipeline()
    from .rag.clause_rag import ClauseRAG
    from .extract.llm_extractor import LLMRuleExtractor
    from .review.pipeline import ReviewPipeline
    from .llm.client import get_client
    rag = ClauseRAG.load()
    c = rag.get_clause(args.clause)
    if c is None:
        print(f"未找到條文: {args.clause}", file=sys.stderr)
        sys.exit(1)
    client = get_client()
    candidates = LLMRuleExtractor(client).extract_clause(c)
    store = {cc.clause_id: cc for cc in rag.clauses}
    pipeline = ReviewPipeline(store)
    reviewed = [pipeline.review_rule(r) for r in candidates]
    _print({
        "backend": client.backend,
        "clause_id": c.clause_id, "text": c.clean_text,
        "llm_candidate_rules": len(candidates),
        "rules": [{"id": r.initial_rule_id, "type": r.rule_type,
                   "if": r.if_conditions, "then": r.then_conclusions,
                   "strength": r.prescription_strength,
                   "release": r.autonomous_review.release_level,
                   "evidence_verified": r.autonomous_review.evidence_verified,
                   "critic_flags": r.autonomous_review.critic_flags}
                  for r in reviewed],
    })


def cmd_tool_call(args):
    _need_pipeline()
    import json as _json
    from .integrations.tool_specs import dispatch
    try:
        arguments = _json.loads(args.args) if args.args else {}
    except _json.JSONDecodeError as exc:
        print(f"--args 不是合法 JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    _print(dispatch(args.name, arguments))


def cmd_export_tools(args):
    from pathlib import Path
    from .integrations.tool_specs import export_specs
    out = export_specs(Path(args.out))
    print(f"tool specs: {out}")


def cmd_serve_mcp(args):
    from .integrations.mcp_server import serve
    serve()


def cmd_serve(args):
    from .server.http_server import serve
    serve(host=args.host, port=args.port, warm=not args.no_warm)


def cmd_ingest(args):
    from .corpus import downloader
    path = downloader.run()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"manifest: {path}")
    print(f"books: {manifest['book_count']}")
    for b in manifest["books"]:
        if b["book_dir"] in ([config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK]
                             + config.VARIANT_BOOKS + [config.COMMENTARY_ALIGN_BOOK]):
            print(f"  [{b['hermes_layer']}] {b['title']} ({b['book_dir']})")


def cmd_library(args):
    from .corpus import library
    act = args.action
    if act == "fetch":
        library.fetch(url=args.url or None, force=args.force,
                      sha256=getattr(args, "sha256", "") or "")
        print(json.dumps(library.status(), ensure_ascii=False, indent=1))
        return
    if act == "status":
        print(json.dumps(library.status(), ensure_ascii=False, indent=1))
        return
    if not library.is_available():
        print("全庫未就緒：請先運行 `python3 -m hermes_shanghan library fetch`",
              file=sys.stderr)
        sys.exit(1)
    lib = library.Library()
    if act == "categories":
        for cat, n in lib.categories().items():
            print(f"{cat or '(未分類)'}\t{n}")
    elif act == "search":
        for h in lib.search(args.query, category=args.category, limit=args.limit):
            subs = f" ⊃{len(h['sub_books'])}子書" if h["sub_books"] else ""
            print(f"{h['id']}\t{h['author']}·{h['dynasty']}\t"
                  f"[{h['category']}] ~{h['approx_chars']}字{subs}")
    elif act == "grep":
        out = lib.grep(args.query, category=args.category, limit=args.limit)
        if "error" in out:
            print(out["error"], file=sys.stderr)
            sys.exit(1)
        for h in out["hits"]:
            print(f"《{h['title']}》{h['author']}·{h['dynasty']} "
                  f"§{h['section']}\n  …{h['excerpt']}…")
        note = "（掃描達上限，命中未必完整）" if out["scan_capped"] else ""
        print(f"-- {out['n_hits']} 命中 / 候選 {out['n_candidate_books']} 部 / "
              f"實掃 {out['n_books_scanned']} 部{note}")
    elif act == "toc":
        for t in lib.toc(args.query):
            print("  " * (t["level"] - 1) + t["title"] + f"  ({t['file']})")
    elif act == "read":
        out = lib.read(args.query, section=args.section,
                       max_chars=args.limit if args.limit > 100 else 4000)
        if "error" in out:
            print(out["error"], file=sys.stderr)
            sys.exit(1)
        b = out["book"]
        print(f"《{b['title']}》{b['author']}·{b['dynasty']}"
              + (f" §{out['section']}" if out["section"] else ""))
        print(out["text"])
        if out["truncated"]:
            print(f"…（截斷，全文 {out['total_chars']} 字）")


def _load_research_or_exit(name: str):
    import json as _json
    p = config.RESEARCH_DIR / name
    if not p.exists():
        print(f"缺少 {p.name}：請先運行 `python3 -m hermes_shanghan pipeline`",
              file=sys.stderr)
        sys.exit(1)
    return _json.loads(p.read_text(encoding="utf-8"))


def cmd_dose(args):
    _need_pipeline()
    ratios = _load_research_or_exit("dose_ratios.json")
    evo = _load_research_or_exit("dose_family_evolution.json")
    if args.formula:
        f = next((x for x in ratios["formulas"] if x["formula"] == args.formula), None)
        if not f:
            print(f"無劑量數據：{args.formula}", file=sys.stderr)
            sys.exit(1)
        _print(f)
        edges = [e for e in evo["edges"]
                 if args.formula in (e["base"], e["modified"]) and e["dose_deltas"]]
        if edges:
            _print({"dose_evolution_edges": edges})
    else:
        _print(_load_research_or_exit("dose_summary.json"))


def cmd_divergence(args):
    _need_pipeline()
    a = _load_research_or_exit("commentary_divergence.json")
    if args.clause:
        rows = [r for r in a["clauses"] if args.clause in r["clause_id"]]
        _print({"book_coverage": a["book_coverage"], "clauses": rows})
    else:
        _print({k: a[k] for k in ("n_books", "n_commentary_rules",
                                  "n_clauses_multi_commentator",
                                  "mean_term_divergence", "book_coverage",
                                  "top_divergent_clauses", "agreement_matrix",
                                  "commentator_fingerprints")})


def cmd_trace(args):
    _need_pipeline()
    from .trace.chains import trace_dispatch
    out = trace_dispatch(args.type, args.ref)
    _print(safety.governed(out, "researcher"))


def cmd_trace_build(args):
    _need_pipeline()
    from .trace.builder import build_all
    _print(build_all(verbose=True))


def cmd_trace_network(args):
    _need_pipeline()
    from .agent.tools import get_registry
    _print(get_registry().call("shanghan_citation_network",
                               {"target": args.target, "top_k": args.top_k,
                                "scope": args.scope}))


def cmd_trace_scan_full(args):
    """段落級全量引文邊導出（~4 萬條，不作為提交資產）。"""
    _need_pipeline()
    from pathlib import Path

    from .schemas import write_jsonl
    from .trace.builder import _clause_texts
    from .trace.quotation import scan_corpus
    commentary = read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
    scan = scan_corpus(_clause_texts(), commentary, verbose=True)
    out = Path(args.out)
    n = write_jsonl(out, scan["edges"])
    _print({"out": str(out), "n_edges": n, "params": scan["params"]})


def cmd_trace_audit_citation(args):
    """引文邊審計（A2）：某書 × 某條文的全部引文邊逐條可靠性核查。"""
    _need_pipeline()
    from .schemas import read_jsonl as _rj
    from .trace.builder import _clause_texts
    from .trace.chains import _clauses, _resolve_clause
    from .trace.quotation import audit_citation
    c = _resolve_clause(args.clause, _clauses())
    if c is None:
        print(f"未找到條文 {args.clause}", file=sys.stderr)
        sys.exit(1)
    commentary = _rj(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
    _print(audit_citation(args.book, c["clause_id"], _clause_texts(), commentary))


def cmd_trace_gold_sample(args):
    """金標準標註表生成（A3）：確定性抽樣（等距/分層）+ 算法預測列。"""
    _need_pipeline()
    from .trace.goldset import build_sample
    _print(build_sample(n=args.n, out_path=args.out, stratify=args.stratify))


def cmd_trace_gold_eval(args):
    """金標準評估（A3）：讀回人工標註，計 P/R/F1 與模式一致率。"""
    _need_pipeline()
    from .trace.goldset import evaluate
    _print(evaluate(args.file))


def cmd_herb(args):
    """藥證檔案（C10）：單味藥的方劑/條文/劑量/配伍畫像。"""
    _need_pipeline()
    from .apps.herbal import herb_profile
    _print(safety.governed(herb_profile(args.name), args.role or "doctor"))


def cmd_formula_explain(args):
    """方解檔案（C11）：源流+方證+鑒別+禁忌+煎服一站式。"""
    _need_pipeline()
    from .trace.chains import formula_explain
    _print(safety.governed(formula_explain(args.name), args.role or "doctor"))


def cmd_run(args):
    """Harness 運行：顯式節點圖 + checkpoint + span 軌跡 + 發布閘門。"""
    _need_pipeline()
    from .agent.harness import HarnessRunner
    st = HarnessRunner().start(args.query, mode=args.mode, role=args.role or
                               ("doctor" if args.mode != "deep-research"
                                else "researcher"),
                               max_steps=args.max_steps,
                               max_tool_calls=args.max_tool_calls)
    _print({"run_id": st.spec.run_id, "status": st.status,
            "release": st.release, "pending_review": st.pending_review,
            "approval_requests": st.approval_requests,
            "answer": st.final_answer,
            "nodes": {k: v.status for k, v in st.nodes.items()},
            "n_tool_calls": len(st.tool_calls),
            "budget": st.budget_snapshot,
            "hint": (f"人工審核：python3 -m hermes_shanghan run-resume "
                     f"{st.spec.run_id} --approve（或 --reject）"
                     if st.status == "paused" else "")})


def cmd_run_list(args):
    from .agent.harness import list_runs
    _print({"runs": list_runs()})


def cmd_run_resume(args):
    _need_pipeline()
    from .agent.harness import HarnessRunner
    st = HarnessRunner().resume(args.run_id, approve=args.approve,
                                reject=args.reject, approver=args.approver)
    if st is None:
        print(f"未找到 run {args.run_id}", file=sys.stderr)
        sys.exit(1)
    _print({"run_id": st.spec.run_id, "status": st.status,
            "release": st.release, "answer": st.final_answer})


def cmd_run_replay(args):
    _need_pipeline()
    from .agent.harness import HarnessRunner
    out = HarnessRunner().replay(args.run_id)
    if out is None:
        print(f"未找到 run {args.run_id}", file=sys.stderr)
        sys.exit(1)
    _print(out)


def cmd_run_export(args):
    from .agent.harness.runner import export_run
    out = export_run(args.run_id, fmt=args.format)
    if out is None:
        print(f"未找到 run {args.run_id}", file=sys.stderr)
        sys.exit(1)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"已導出 {args.out}")
    else:
        print(out)


def cmd_readyz(args):
    """就緒探針：數據能力逐項校驗（拒絕假健康）。exit 0=ready, 2=not。"""
    from .health import readyz
    out = readyz(include_runtime=args.runtime)
    _print(out)
    sys.exit(0 if out["ready"] else 2)


def cmd_intake(args):
    """四診信息採集：自然敘述 → 結構化四診表（信息整理，非診斷）。"""
    _need_pipeline()
    from .apps.bianzheng import intake_parse
    _print(safety.governed(intake_parse(args.text), args.role or "patient"))


def cmd_adjudicate(args):
    """方證多假設裁決（醫師/教學端）。"""
    _need_pipeline()
    from .apps.bianzheng import adjudicate
    out = adjudicate([s for s in args.symptoms.split(",") if s.strip()],
                     pulse=[p for p in (args.pulse or "").split(",") if p.strip()],
                     six_channel=args.six_channel)
    _print(safety.governed(out, args.role or "doctor"))


def cmd_conflict_check(args):
    """方證衝突審計（醫師端）。"""
    _need_pipeline()
    from .apps.bianzheng import conflict_audit
    out = conflict_audit(args.formula,
                         [s for s in args.symptoms.split(",") if s.strip()])
    _print(safety.governed(out, args.role or "doctor"))


def cmd_simulate_mistreatment(args):
    """誤治傳變路徑模擬。"""
    _need_pipeline()
    from .apps.bianzheng import mistreatment_simulate
    _print(mistreatment_simulate(args.channel, args.type, steps=args.steps))


def cmd_trace_audit_scope(args):
    """Scope 一致性審計（A1）：三個 scope 的計量輸出逐一遞歸掃描違例。"""
    _need_pipeline()
    from .agent.tools import get_registry
    from .trace.scientometrics import audit_scope_consistency
    reg = get_registry()
    reports = []
    for scope in ("canonical", "auxiliary", "all"):
        payload = reg.call("shanghan_citation_network",
                           {"scope": scope, "top_k": 20})
        reports.append(audit_scope_consistency(payload, scope))
    ok = all(r["ok"] for r in reports)
    _print({"ok": ok, "reports": reports,
            "note": "canonical 輸出含 AUX 或 auxiliary 輸出含正文條文即為違例；"
                    "對工具輸出全文遞歸掃描，杜絕漏檢字段。"})
    if not ok:
        sys.exit(1)


def cmd_trace_scan_library(args):
    """全庫（中醫笈成 800+ 部）引文掃描；庫未下載時如實提示。"""
    _need_pipeline()
    from pathlib import Path

    from .schemas import write_jsonl
    from .trace.builder import _clause_texts
    from .trace.quotation import scan_library
    res = scan_library(_clause_texts(), category=args.category,
                       limit=args.limit, verbose=True)
    if not res.get("available"):
        _print(res)
        sys.exit(1)
    payload = {k: res[k] for k in ("available", "n_units_scanned",
                                   "n_edges", "note")}
    payload["top_books"] = res["book_stats"][:20]
    if args.out:
        n = write_jsonl(Path(args.out), res["edges"])
        payload["out"] = args.out
        payload["n_written"] = n
    _print(payload)


def cmd_solve(args):
    _need_pipeline()
    from .agent.complex_agent import ComplexAgent
    out = ComplexAgent().solve(args.question, role=args.role)
    if args.answer_only:
        print(out.get("answer", out.get("message", "")))
    else:
        _print(out)


def cmd_deep_research(args):
    _need_pipeline()
    from .agent.research_loop import DeepResearcher
    dossier = DeepResearcher(max_rounds=args.rounds).run(args.topic)
    _print(safety.governed(dossier, "researcher"))


def cmd_evaluate(args):
    _need_pipeline()
    from .eval.runner import run_suites
    suites = tuple(args.suite.split(",")) if args.suite != "all" \
        else ("cloze", "cases", "grounding", "agent")
    summary = run_suites(suites=suites, ablations=args.ablations,
                         limit=args.limit)
    _print(summary)


def cmd_stats(args):
    _need_pipeline()
    from collections import Counter
    rules = read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
    clauses = read_jsonl(config.CLAUSE_DIR / "clauses.jsonl")
    rels = read_jsonl(config.RELATION_DIR / "clause_relations.jsonl")
    levels = Counter(r["autonomous_review"]["release_level"] for r in rules)
    types = Counter(r["rule_type"] for r in rules)
    _print({
        "clauses": len(clauses),
        "canonical": sum(1 for c in clauses if c["text_type"] == "original_clause"),
        "initial_rules": len(rules),
        "release_levels": dict(levels),
        "rule_types": dict(types.most_common()),
        "clause_relations": len(rels),
        "formula_pattern_rules": len(read_jsonl(config.RULES_FORMULA_DIR / "formula_pattern_rules.jsonl")),
        "six_channel_rules": len(read_jsonl(config.RULES_SIX_CHANNEL_DIR / "six_channel_rules.jsonl")),
        "therapy_rules": len(read_jsonl(config.RULES_THERAPY_DIR / "therapy_rules.jsonl")),
        "mistreatment_rules": len(read_jsonl(config.RULES_MISTREATMENT_DIR / "mistreatment_rules.jsonl")),
        "differential_rules": len(read_jsonl(config.RULES_DIFFERENTIAL_DIR / "differential_rules.jsonl")),
        "merged_rules": len(read_jsonl(config.RULES_MERGED_DIR / "merged_rules.jsonl")),
        "variant_rules": len(read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl")),
        "commentary_rules": len(read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")),
        "rejected": len(read_jsonl(config.REJECTED_DIR / "rejected_rules.jsonl")),
        "audits": len(read_jsonl(config.AUDIT_DIR / "audit_log.jsonl")),
    })


def cmd_search(args):
    _need_pipeline()
    from .rag.clause_rag import ClauseRAG
    rag = ClauseRAG.load()
    hits = rag.search(args.query, top_k=args.top_k,
                      six_channel=args.six_channel, formula=args.formula,
                      field=args.field, expand_relations=args.expand)
    _print({"query": args.query, "hits": hits})


def cmd_clause(args):
    _need_pipeline()
    from .rag.clause_rag import ClauseRAG
    rag = ClauseRAG.load()
    c = rag.get_clause(args.ref)
    if c is None:
        print(f"未找到條文: {args.ref}", file=sys.stderr)
        sys.exit(1)
    rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
             if r["clause_id"] == c.clause_id]
    _print({
        "clause": c.to_dict(),
        "initial_rules": [{
            "id": r["initial_rule_id"], "type": r["rule_type"],
            "strength": r.get("prescription_strength", ""),
            "release": r["autonomous_review"]["release_level"]} for r in rules],
        "relations": rag.related(c.clause_id),
    })


def cmd_explain_clause(args):
    _need_pipeline()
    from .rag.clause_rag import ClauseRAG
    rag = ClauseRAG.load()
    c = rag.get_clause(args.ref)
    if c is None:
        print(f"未找到條文: {args.ref}", file=sys.stderr)
        sys.exit(1)
    rules = [r for r in read_jsonl(config.RULES_INITIAL_DIR / "initial_rules.jsonl")
             if r["clause_id"] == c.clause_id]
    variants = [v for v in read_jsonl(config.RULES_VARIANT_DIR / "variant_rules.jsonl")
                if v["clause_id"] == c.clause_id]
    comments = [v for v in read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
                if v["clause_id"] == c.clause_id]
    payload = {
        "clause_id": c.clause_id,
        "original_text": {"layer": "A 原文直述", "text": c.clean_text,
                          "chapter": c.chapter, "six_channel": c.six_channel,
                          "clause_number": c.clause_number},
        "entities": {"symptoms": c.symptoms, "negated_findings": c.negated_findings,
                     "pulse": c.pulse, "formulas": c.formula_names,
                     "disease_patterns": c.disease_patterns,
                     "therapy": c.therapy_terms,
                     "contraindications": c.contraindication_terms,
                     "mistreatment": c.mistreatment_terms,
                     "prognosis": c.prognosis_terms},
        "formula_blocks": [fb.to_dict() for fb in c.formula_blocks],
        "initial_rules": rules,
        "relations": rag.related(c.clause_id, limit=10),
        "variants": [{"layer": "B 版本異文", **{k: v[k] for k in
                      ("variant_book", "variant_text", "similarity", "notable_differences")}}
                     for v in variants],
        "commentaries": [{"layer": "C 注家解釋", "commentator": v["commentator"],
                          "text": v["commentary_text"][:300]} for v in comments],
        "model_reading": {"layer": "E 模型推理",
                          "note": "以上實體標註與規則由模型流水線生成，已經自主審核分級。"},
    }
    _print(safety.governed(payload, args.role))


def cmd_match(args):
    _need_pipeline()
    from .apps.doctor import FormulaMatcher
    from .orchestrator import Artifacts
    art = Artifacts()
    matcher = FormulaMatcher(art.formula_rules, art.clause_store())
    res = matcher.match(symptoms=args.symptoms.split(",") if args.symptoms else [],
                        pulse=args.pulse.split(",") if args.pulse else [],
                        six_channel=args.six_channel, top_k=args.top_k)
    _print(res)


def cmd_ask(args):
    _need_pipeline()
    from .rag.skill_rag import SkillRAG
    from .orchestrator import Artifacts
    rag = SkillRAG()
    route = rag.route(args.question, role=args.role)
    art = Artifacts()
    payload = {"question": args.question, "routing": route}

    handler = route["handler"]
    role = route["role"]
    if handler == "patient" or role == "patient":
        from .apps.patient import PatientEducator
        edu = PatientEducator(art.six_channel_rules, art.clause_store())
        _print(edu.explain(args.question))
        return
    if handler == "clause":
        import re as _re
        m = _re.search(r"(\d{1,3})", args.question)
        if m:
            args2 = argparse.Namespace(ref=m.group(1), role=role)
            cmd_explain_clause(args2)
            return
    if handler == "six_channel":
        from .apps.teaching import TeachingBuilder
        channel = next((c for c in list(config.CHANNEL_PINYIN) if c in args.question
                        or c in args.question.replace("阳", "陽").replace("阴", "陰")), None)
        if channel:
            tb = TeachingBuilder(art.clauses, art.six_channel_rules,
                                 art.formula_rules, art.mistreatment_rules)
            _print(tb.lesson(channel))
            return
    if handler == "differential":
        from . import lexicon as _lx
        names = [n for n in sorted(_lx.FORMULA_SEEDS, key=len, reverse=True)
                 if n in args.question][:3]
        if len(names) >= 2:
            diffs = [d for d in art.differential_rules
                     if set(names) <= set(d.formulas)]
            if diffs:
                _print(safety.governed({"question": args.question,
                                        "differential": diffs[0].to_dict()}, role))
                return
    # generic: clause RAG with evidence chain + skill rules
    from .rag.clause_rag import ClauseRAG
    crag = ClauseRAG.load()
    hits = crag.search(args.question, top_k=5, expand_relations=True)
    payload["evidence"] = hits
    payload["skill_rules_sample"] = rag.skill_rules(route["skill"], limit=3)
    payload["answer_protocol"] = "無條文編號，不成證據——以上每條證據均帶 clause_id。"
    _print(safety.governed(payload, role))


def cmd_teach(args):
    _need_pipeline()
    from .apps.teaching import TeachingBuilder
    from .orchestrator import Artifacts
    art = Artifacts()
    tb = TeachingBuilder(art.clauses, art.six_channel_rules,
                         art.formula_rules, art.mistreatment_rules)
    _print(tb.lesson(args.channel))


def cmd_differential(args):
    _need_pipeline()
    from .orchestrator import Artifacts
    from .textutil import normalize_query
    art = Artifacts()
    names = [normalize_query(f) for f in args.formulas]
    cands = [d for d in art.differential_rules if set(names) <= set(d.formulas)]
    if not cands:
        cands = [d for d in art.differential_rules
                 if len(set(names) & set(d.formulas)) >= 2]
    if not cands:
        from .induce.differential import DifferentialInducer
        ind = DifferentialInducer(art.formula_rules)
        one = ind._build_one(names, 999)
        cands = [one] if one else []
    if not cands:
        print("未能構建該鑒別對（方證規則缺失）", file=sys.stderr)
        sys.exit(1)
    _print(safety.governed({"differential": cands[0].to_dict()}, "doctor"))


def cmd_research(args):
    _need_pipeline()
    from .apps.research import ResearchMiner
    from .orchestrator import Artifacts
    art = Artifacts()
    miner = ResearchMiner(art.clauses, art.formula_rules, art.mistreatment_rules)
    res = miner.run_topic(args.topic, outputs=args.outputs.split(","))
    _print(res)


def cmd_paper(args):
    _need_pipeline()
    from .paper.writer import PaperWriter
    from .orchestrator import Artifacts
    art = Artifacts()
    writer = PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                         art.six_channel_rules, art.mistreatment_rules,
                         art.differential_rules, commentary_rules=art.commentary_rules)
    path = writer.generate(paper_type=args.type, topic=args.topic or "",
                           use_llm=not args.no_llm)
    print(f"manuscript: {path}")


def cmd_skills(args):
    from .rag.skill_rag import SkillRAG
    rag = SkillRAG()
    for s in rag.describe():
        print(f"{s['name']:48s} {s['description'][:60]}")


def cmd_visit_summary(args):
    _need_pipeline()
    from .apps.patient import PatientEducator
    from .orchestrator import Artifacts
    art = Artifacts()
    edu = PatientEducator(art.six_channel_rules, art.clause_store())
    _print(edu.organize_symptoms(args.symptoms.split(",")))


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="hermes-shanghan",
                                description="《傷寒論》自主規則挖掘與 Skill 生成系統")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pipeline", help="運行全部工作流")
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--llm-extract", action="store_true",
                    help="啟用 LLM 抽取增強（候選規則仍過全部審核閘門）")
    sp.add_argument("--llm-critic", action="store_true",
                    help="啟用 LLM 對抗式批評器作為附加審核閘門")
    sp.set_defaults(func=cmd_pipeline)

    sp = sub.add_parser("llm-status", help="查看 LLM 後端狀態與配置")
    sp.set_defaults(func=cmd_llm_status)

    sp = sub.add_parser("agent", help="智能體問答（工具調用+回源核驗+安全治理）")
    sp.add_argument("question")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.add_argument("--max-steps", type=int, default=5)
    sp.add_argument("--answer-only", action="store_true")
    sp.set_defaults(func=cmd_agent)

    sp = sub.add_parser("llm-extract", help="LLM 抽取單條規則並過審核閘門")
    sp.add_argument("clause", help="條文號或 clause_id")
    sp.set_defaults(func=cmd_llm_extract)

    sp = sub.add_parser("tool-call", help="直接調用一個工具（harness 分發目標）")
    sp.add_argument("name")
    sp.add_argument("--args", default="{}", help="JSON 參數")
    sp.set_defaults(func=cmd_tool_call)

    sp = sub.add_parser("export-tools", help="導出 OpenAI/Anthropic 工具規格")
    sp.add_argument("--out", default="data/shanghan/tool_specs.json")
    sp.set_defaults(func=cmd_export_tools)

    sp = sub.add_parser("serve-mcp", help="啟動 MCP stdio 服務器（Claude Code 等）")
    sp.set_defaults(func=cmd_serve_mcp)

    sp = sub.add_parser("serve", help="啟動 Web 控制台 UI（集成全部功能）")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-warm", action="store_true", help="不預熱（首個請求較慢）")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("ingest", help="語料導入與 manifest")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("library",
                        help="中醫笈成全庫（800+ 部）：自動下載/編目檢索/全文查閱")
    sp.add_argument("action", choices=["fetch", "status", "search", "grep",
                                       "toc", "read", "categories"])
    sp.add_argument("query", nargs="?", default="",
                    help="檢索詞（search/grep）或書名（toc/read）")
    sp.add_argument("--category", default="", help="按分類過濾，如 醫案/本草/溫病")
    sp.add_argument("--section", default="", help="read：只讀某一章節")
    sp.add_argument("--limit", type=int, default=12)
    sp.add_argument("--url", default="",
                    help="fetch：覆蓋默認庫源 URL（須 HERMES_LIBRARY_ALLOW_CUSTOM=1"
                         " 並提供 --sha256，fail-closed）")
    sp.add_argument("--sha256", default="",
                    help="fetch：自定義庫源的 SHA-256（64 位十六進制，必填）")
    sp.add_argument("--force", action="store_true", help="fetch：強制重建")
    sp.set_defaults(func=cmd_library)

    sp = sub.add_parser("dose", help="劑量計量層：藥量比/折算/家族劑量演化")
    sp.add_argument("formula", nargs="?", default="")
    sp.set_defaults(func=cmd_dose)

    sp = sub.add_parser("divergence", help="注家分歧圖譜：覆蓋/爭點條文/一致度矩陣")
    sp.add_argument("--clause", default="", help="按 clause_id 片段過濾")
    sp.set_defaults(func=cmd_divergence)

    sp = sub.add_parser("trace",
                        help="深度溯源鏈：條文/方劑/方證觀點/注家/學派/任意文本回源")
    sp.add_argument("ref", help="條文號、方名、觀點關鍵詞、注家名、學派名或原文片段")
    sp.add_argument("--type", "-t", default="text",
                    choices=["clause", "formula", "claim", "school",
                             "commentator", "text", "quote", "term",
                             "dispute", "compare", "argument"],
                    help="溯源對象類型（默認 text：任意文本回源；"
                         "quote：誤引檢測；term：術語譜系；dispute：注家爭議結構化；"
                         "compare：學派/注家比較（A vs B）；"
                         "argument：方證論證結構（支持/反證/異文分叉/注家共同與"
                         "爭議/隱含假設/不可裁決，七段分層））")
    sp.set_defaults(func=cmd_trace)

    sp = sub.add_parser("trace-build", help="重建溯源層資產（引文邊/計量網絡/學派/觀點）")
    sp.set_defaults(func=cmd_trace_build)

    sp = sub.add_parser("trace-network", help="學術計量網絡：引文/共引/耦合/切片/突現/主路徑")
    sp.add_argument("--target", default="", help="可選：條文號或方名")
    sp.add_argument("--scope", default="canonical",
                    choices=["canonical", "auxiliary", "all"],
                    help="被引榜範圍：正文398條（默認）/輔助篇章/混排")
    sp.add_argument("--top-k", type=int, default=8)
    sp.set_defaults(func=cmd_trace_network)

    sp = sub.add_parser("trace-scan-full",
                        help="導出段落級全量引文邊 jsonl（~4 萬條，體積大不入庫）")
    sp.add_argument("--out", required=True, help="輸出 jsonl 路徑")
    sp.set_defaults(func=cmd_trace_scan_full)

    sp = sub.add_parser("trace-audit-scope",
                        help="Scope 一致性審計：canonical/auxiliary/all 輸出逐一驗證無跨域混入")
    sp.set_defaults(func=cmd_trace_audit_scope)

    sp = sub.add_parser("trace-audit-citation",
                        help="引文邊審計：某書×某條文逐邊可靠性核查（片段/覆蓋/歧義/轉引）")
    sp.add_argument("--book", required=True, help="書目錄名，如 傷寒來蘇集")
    sp.add_argument("--clause", required=True, help="條文號或 clause_id")
    sp.set_defaults(func=cmd_trace_audit_citation)

    sp = sub.add_parser("trace-gold-sample",
                        help="引文識別金標準：確定性抽樣導出標註表 CSV（附算法預測列）")
    sp.add_argument("--n", type=int, default=50)
    sp.add_argument("--out", required=True, help="輸出 CSV 路徑")
    sp.add_argument("--stratify", action="store_true",
                    help="分層抽樣（朝代×預測模式，論文級評測用；默認等距）")
    sp.set_defaults(func=cmd_trace_gold_sample)

    sp = sub.add_parser("trace-gold-eval",
                        help="引文識別金標準：讀回人工標註計 P/R/F1 與模式一致率")
    sp.add_argument("--file", required=True, help="已標註 CSV 路徑")
    sp.set_defaults(func=cmd_trace_gold_eval)

    sp = sub.add_parser("run",
                        help="Harness 運行：節點圖+checkpoint+span 軌跡+發布閘門（可恢復）")
    sp.add_argument("query")
    sp.add_argument("--mode", default="agent",
                    choices=["agent", "council", "deep-research", "solve"])
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.add_argument("--max-steps", type=int, default=6)
    sp.add_argument("--max-tool-calls", type=int, default=12,
                    help="統一工具預算（Harness 控制器原子扣減，批量調用不可突破）")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("run-list", help="列出 harness 運行")
    sp.set_defaults(func=cmd_run_list)

    sp = sub.add_parser("run-resume",
                        help="恢復中斷的運行 / --approve 批准（重跑下游閘門後放行）"
                             " / --reject 駁回")
    sp.add_argument("run_id")
    sp.add_argument("--approve", action="store_true")
    sp.add_argument("--reject", action="store_true")
    sp.add_argument("--approver", default="")
    sp.set_defaults(func=cmd_run_resume)

    sp = sub.add_parser("run-replay", help="重放運行（local 後端全確定，對比回答指紋）")
    sp.add_argument("run_id")
    sp.set_defaults(func=cmd_run_replay)

    sp = sub.add_parser("run-export", help="導出運行報告（md/json：節點軌跡+工具span+證據台賬）")
    sp.add_argument("run_id")
    sp.add_argument("--format", default="md", choices=["md", "json"])
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_run_export)

    sp = sub.add_parser("readyz",
                        help="就緒探針：manifest/398條/規則庫/索引/工具規格逐項"
                             "校驗（拒絕假健康空運行；exit 2=未就緒）")
    sp.add_argument("--runtime", action="store_true",
                    help="額外對比運行時工具註冊表與提交的 tool_specs.json")
    sp.set_defaults(func=cmd_readyz)

    sp = sub.add_parser("intake",
                        help="四診信息採集：自然敘述→結構化四診表+缺失信息+追問（非診斷）")
    sp.add_argument("text")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_intake)

    sp = sub.add_parser("adjudicate",
                        help="方證多假設裁決：三態裁決+為什麼還不能定方+關鍵追問")
    sp.add_argument("--symptoms", required=True, help="逗號分隔")
    sp.add_argument("--pulse", default="", help="逗號分隔")
    sp.add_argument("--six-channel", default="")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_adjudicate)

    sp = sub.add_parser("conflict-check",
                        help="方證衝突審計：衝突項/衝突條文/是否禁忌/改判候選/應補問")
    sp.add_argument("--formula", required=True)
    sp.add_argument("--symptoms", required=True, help="逗號分隔")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_conflict_check)

    sp = sub.add_parser("simulate-mistreatment",
                        help="誤治傳變路徑模擬：經×誤治→變證→救逆方→條文依據")
    sp.add_argument("--channel", default="太陽病")
    sp.add_argument("--type", default="", help="誤汗/誤下/誤吐/火逆；留空列全部")
    sp.add_argument("--steps", type=int, default=1)
    sp.set_defaults(func=cmd_simulate_mistreatment)

    sp = sub.add_parser("herb", help="藥證檔案：方劑/條文/劑量寫法/配伍網絡（不編造藥性解釋）")
    sp.add_argument("name")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_herb)

    sp = sub.add_parser("formula-explain",
                        help="方解檔案：首見/方證/組成劑量/煎服/禁忌/類方鑒別/方名傳播一站式")
    sp.add_argument("name")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_formula_explain)

    sp = sub.add_parser("trace-scan-library",
                        help="全庫引文掃描（中醫笈成 800+ 部，旁證層；需先 library fetch）")
    sp.add_argument("--category", default="", help="分類過濾：本草/方書/醫案/溫病…")
    sp.add_argument("--limit", type=int, default=0, help="只掃前 N 個文本單元（0=全部）")
    sp.add_argument("--out", default="", help="可選：導出邊 jsonl 路徑")
    sp.set_defaults(func=cmd_trace_scan_library)

    sp = sub.add_parser("solve", help="複合問題編排：任務分解→作用域子代理→綜合核驗")
    sp.add_argument("question")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.add_argument("--answer-only", action="store_true")
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("deep-research", help="深度研究循環：規劃→子代理取證→批評家→溯源檔案")
    sp.add_argument("topic")
    sp.add_argument("--rounds", type=int, default=3)
    sp.set_defaults(func=cmd_deep_research)

    sp = sub.add_parser("evaluate",
                        help="客觀評測：遮方預測/醫案回放/證據接地率/智能體基準")
    sp.add_argument("--suite", default="all",
                    help="all 或逗號分隔：cloze,cases,grounding,agent")
    sp.add_argument("--ablations", action="store_true",
                    help="對匹配器各評分組件做消融實驗")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("stats", help="規則庫統計")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("search", help="原文 RAG 檢索")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=8)
    sp.add_argument("--six-channel")
    sp.add_argument("--formula")
    sp.add_argument("--field", choices=["symptom", "pulse", "therapy",
                                        "contraindication", "mistreatment",
                                        "formula", "disease"])
    sp.add_argument("--expand", action="store_true", help="關係圖譜擴展")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("clause", help="按條文號/ID 查看條文")
    sp.add_argument("ref")
    sp.set_defaults(func=cmd_clause)

    sp = sub.add_parser("explain-clause", help="條文全息解釋（原文/異文/注/規則/關係）")
    sp.add_argument("ref")
    sp.add_argument("--role", default="student", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_explain_clause)

    sp = sub.add_parser("match", help="醫師端方證匹配")
    sp.add_argument("--symptoms", required=True, help="逗號分隔，如: 惡寒,發熱,無汗,身疼痛")
    sp.add_argument("--pulse", default="", help="逗號分隔，如: 浮緊")
    sp.add_argument("--six-channel")
    sp.add_argument("--top-k", type=int, default=5)
    sp.set_defaults(func=cmd_match)

    sp = sub.add_parser("ask", help="Skill RAG 問答（自動路由角色與技能）")
    sp.add_argument("question")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("teach", help="六經教學")
    sp.add_argument("channel", help="如: 太陽病 / 太陽 / 少陰病")
    sp.set_defaults(func=cmd_teach)

    sp = sub.add_parser("differential", help="方證鑒別")
    sp.add_argument("formulas", nargs="+")
    sp.set_defaults(func=cmd_differential)

    sp = sub.add_parser("research", help="科研挖掘")
    sp.add_argument("topic")
    sp.add_argument("--outputs", default="rules,network,paper_outline")
    sp.set_defaults(func=cmd_research)

    sp = sub.add_parser("paper", help="自動論文生成")
    sp.add_argument("--type", default="formula_pattern",
                    choices=["formula_pattern", "six_channel_kg", "mistreatment",
                             "network_pharmacology", "commentary_compare",
                             "methodology", "benchmark", "provenance"])
    sp.add_argument("--topic", default="")
    sp.add_argument("--no-llm", action="store_true",
                    help="跳過增益層起草，只輸出確定性模板與數據表格")
    sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("skills", help="列出已編譯 Skill")
    sp.set_defaults(func=cmd_skills)

    sp = sub.add_parser("visit-summary", help="患者端：就診症狀清單整理（不做任何判斷）")
    sp.add_argument("--symptoms", required=True, help="逗號分隔的自述症狀")
    sp.set_defaults(func=cmd_visit_summary)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
