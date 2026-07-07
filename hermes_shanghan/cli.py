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
        else ("cloze", "cases", "grounding")
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

    sp = sub.add_parser("dose", help="劑量計量層：藥量比/折算/家族劑量演化")
    sp.add_argument("formula", nargs="?", default="")
    sp.set_defaults(func=cmd_dose)

    sp = sub.add_parser("divergence", help="注家分歧圖譜：覆蓋/爭點條文/一致度矩陣")
    sp.add_argument("--clause", default="", help="按 clause_id 片段過濾")
    sp.set_defaults(func=cmd_divergence)

    sp = sub.add_parser("solve", help="複合問題編排：任務分解→作用域子代理→綜合核驗")
    sp.add_argument("question")
    sp.add_argument("--role", choices=list(safety.ROLES))
    sp.add_argument("--answer-only", action="store_true")
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("deep-research", help="深度研究循環：規劃→子代理取證→批評家→溯源檔案")
    sp.add_argument("topic")
    sp.add_argument("--rounds", type=int, default=3)
    sp.set_defaults(func=cmd_deep_research)

    sp = sub.add_parser("evaluate", help="客觀評測：遮方預測/醫案回放/證據接地率")
    sp.add_argument("--suite", default="all",
                    help="all 或逗號分隔：cloze,cases,grounding")
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
