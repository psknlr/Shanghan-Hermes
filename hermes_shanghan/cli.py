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
    stats = run_pipeline(verbose=not args.quiet)
    _print(stats)


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
                         art.differential_rules)
    path = writer.generate(paper_type=args.type, topic=args.topic or "")
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
    sp.set_defaults(func=cmd_pipeline)

    sp = sub.add_parser("ingest", help="語料導入與 manifest")
    sp.set_defaults(func=cmd_ingest)

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
                             "network_pharmacology", "commentary_compare", "methodology"])
    sp.add_argument("--topic", default="")
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
