"""溯源層（trace）測試：引文模式識別、統一 ID、學派/觀點、計量網絡、
五類溯源鏈、工具接線與可復現性。"""
import json
import unittest

from hermes_shanghan import config


def _ensure_artifacts():
    if not (config.RESEARCH_DIR / "commentary_divergence.json").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)
    from hermes_shanghan.trace.builder import ensure_built
    ensure_built()


# ---------------------------------------------------------------------------
# 引文模式識別（單元級：合成段落，不依賴掃描資產）
# ---------------------------------------------------------------------------
class TestQuotationScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.trace.builder import _clause_texts
        from hermes_shanghan.trace.quotation import QuotationScanner
        cls.texts = _clause_texts()
        cls.scanner = QuotationScanner(cls.texts)

    def test_explicit_quote_detected_as_mingyin(self):
        t = self.texts["SHL_SONGBEN_0001"]
        edges, _ = self.scanner.scan_paragraph(f"仲景曰：{t}。此太陽之綱領也。")
        hit = next(e for e in edges if e["clause_id"] == "SHL_SONGBEN_0001")
        self.assertEqual(hit["mode"], "明引")
        self.assertGreaterEqual(hit["coverage"], 0.7)
        self.assertTrue(hit["marker"])

    def test_unmarked_full_quote_is_anyin(self):
        t = self.texts["SHL_SONGBEN_0012"]
        edges, _ = self.scanner.scan_paragraph(f"蓋{t}，此桂枝湯之正局。")
        hit = next(e for e in edges if e["clause_id"] == "SHL_SONGBEN_0012")
        self.assertEqual(hit["mode"], "暗引")

    def test_fragment_with_marker_is_jieyin(self):
        t = self.texts["SHL_SONGBEN_0016"]
        frag = t[:12]
        edges, _ = self.scanner.scan_paragraph(f"經云{frag}，法當隨證。")
        hit = next(e for e in edges if e["clause_id"] == "SHL_SONGBEN_0016")
        self.assertEqual(hit["mode"], "節引")

    def test_variant_glyphs_still_match(self):
        # 異體字（脅→脇）折疊後仍可回源
        edges, _ = self.scanner.scan_paragraph("仲景曰：往來寒熱，胸脇苦滿，嘿嘿不欲飲食。")
        self.assertTrue(any(e["clause_id"] == "SHL_SONGBEN_0096" for e in edges))

    def test_unresolved_marker_counted_not_fabricated(self):
        # 引《內經》語：庫內無此文 → 只計存疑，不得產生指向條文的邊
        edges, unresolved = self.scanner.scan_paragraph(
            "內經曰：陰陽者天地之道也，萬物之綱紀。")
        self.assertFalse([e for e in edges if e.get("longest_run", 0) >= 8])
        self.assertTrue(unresolved)

    def test_dialogue_marker_excluded(self):
        _, unresolved = self.scanner.scan_paragraph("問曰：何謂也？答曰：未知其詳。")
        self.assertEqual(unresolved, [])

    def test_selfcheck_benchmark(self):
        from hermes_shanghan.trace.quotation import selfcheck
        r = selfcheck(self.texts)
        for mode in ("明引", "節引", "暗引"):
            self.assertGreaterEqual(r["per_mode"][mode]["detection_rate"], 0.9)
            self.assertGreaterEqual(r["per_mode"][mode]["mode_agreement"], 0.9)
        self.assertGreaterEqual(r["per_mode"]["改寫"]["detection_rate"], 0.5)
        self.assertLessEqual(r["negative"]["false_positive_rate"], 0.05)


# ---------------------------------------------------------------------------
# 資產層：統一 ID / 引文邊 / 計量網絡 / 學派 / 觀點
# ---------------------------------------------------------------------------
class TestTraceAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_id_registry(self):
        from hermes_shanghan.trace.builder import load_registry
        reg = load_registry()
        self.assertEqual(reg["counts"]["works"], 57)
        self.assertGreaterEqual(reg["counts"]["formulas"], 100)
        # 朝代補注生效且標記透明
        jiyi = next(w for w in reg["works"] if w["book_dir"] == "傷寒論輯義")
        self.assertEqual(jiyi["dynasty"], "日本")
        self.assertTrue(jiyi["dynasty_overridden"])

    def test_citation_edges_aggregated(self):
        from hermes_shanghan.trace.builder import load_agg_edges
        rows = load_agg_edges()
        self.assertGreater(len(rows), 5000)
        r0 = rows[0]
        for key in ("book_dir", "clause_id", "modes", "max_coverage"):
            self.assertIn(key, r0)
        # A/B 層底本不得出現在引用方
        citing = {r["book_dir"] for r in rows}
        for base in (config.PRIMARY_BOOK, config.SONGBEN_FULL_BOOK,
                     *config.VARIANT_BOOKS):
            self.assertNotIn(base, citing)

    def test_network_metrics(self):
        from hermes_shanghan.trace.builder import load_network
        net = load_network()
        ov = net["overview"]
        self.assertGreater(ov["n_clause_edges"], 10000)
        self.assertGreater(ov["n_citing_works"], 30)
        self.assertTrue(net["top_cited_clauses"])
        self.assertTrue(net["cocitation_pairs"])
        self.assertTrue(net["bibliographic_coupling"])
        # 時間切片按朝代先後排序
        orders = [s["dynasty"] for s in net["time_slices"]]
        self.assertLess(orders.index("宋"), orders.index("清"))
        # 主路徑以原典起點
        mp = net["main_paths"][0]
        self.assertEqual(mp["path"][0]["book"], "傷寒論")

    def test_school_registry_grounded_in_atlas(self):
        from hermes_shanghan.trace.builder import load_schools
        reg = load_schools()
        self.assertEqual(reg["n_schools"], 10)
        cuojian = next(s for s in reg["schools"] if s["school_id"] == "SCH_CUOJIAN")
        self.assertEqual(cuojian["source_level"], "posthoc_induction")
        # 跨派一致度證據已回填（方有執所在派 vs 他派實測分歧）
        self.assertTrue(cuojian["agreement"]["most_divergent_cross_pairs"])

    def test_claims_grading_from_data(self):
        from hermes_shanghan.trace.builder import load_claims
        claims = load_claims()["claims"]
        gzt = next(c for c in claims if c["claim_id"] == "CLAIM_GZT_YINGWEI")
        # 「榮氣和/衛氣不和」逐字見於 53/54 條 → 原文直述成分必須被識別
        self.assertIn("原文直述成分", gzt["evidence_grade"])
        verbatim = gzt["terms_verbatim_in_original"]
        self.assertIn("SHL_SONGBEN_0053", sum(verbatim.values(), []))
        # 注家時間線按朝代排序且非空
        chron = gzt["commentarial_chronology"]
        self.assertTrue(chron)
        orders = [e["dynasty_order"] for e in chron]
        self.assertEqual(orders, sorted(orders))

    def test_rebuild_is_byte_identical(self):
        import hashlib
        from hermes_shanghan.trace.builder import build_all, trace_dir

        def digest():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(trace_dir().glob("*.json*"))}
        before = digest()
        build_all()
        self.assertEqual(before, digest())


# ---------------------------------------------------------------------------
# 五類溯源鏈
# ---------------------------------------------------------------------------
class TestChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_clause_chain(self):
        from hermes_shanghan.trace.chains import clause_chain
        r = clause_chain("12")
        self.assertEqual(r["chain_type"], "原文溯源鏈")
        self.assertEqual(r["clause"]["clause_id"], "SHL_SONGBEN_0012")
        self.assertTrue(r["variants"])            # B 層
        self.assertTrue(r["commentaries"])        # C 層
        self.assertGreater(r["citations"]["n_citing_books"], 10)
        self.assertEqual(r["main_path"][0]["dynasty"], "東漢")
        self.assertIn("A 原文直述", r["evidence_grade"])

    def test_formula_chain(self):
        from hermes_shanghan.trace.chains import formula_chain
        r = formula_chain("桂枝湯")
        self.assertEqual(r["chain_type"], "方劑源流鏈")
        # 首見（宋本條文序）為單一正文條文；支持條文全集正文/輔助分列
        self.assertEqual(r["first_attestation"]["clause_id"], "SHL_SONGBEN_0012")
        self.assertTrue(r["supporting_clauses"]["canonical"])
        self.assertFalse([c for c in r["supporting_clauses"]["canonical"]
                          if "AUX" in c])
        self.assertTrue(r["family_dose_evolution"])
        self.assertGreater(r["name_transmission"]["n_books"], 20)
        self.assertTrue(r["claims"])

    def test_claim_chain_finds_by_keyword(self):
        from hermes_shanghan.trace.chains import claim_chain
        r = claim_chain("營衛不和")
        self.assertEqual(r["chain_type"], "方證觀點演化鏈")
        self.assertEqual(r["formula"], "桂枝湯")
        self.assertTrue(r["commentarial_chronology"])

    def test_school_and_commentator_chains(self):
        from hermes_shanghan.trace.chains import commentator_chain, school_chain
        s = school_chain("錯簡重訂")
        self.assertEqual(s["school_id"], "SCH_CUOJIAN")
        self.assertTrue(s["member_citation_breadth"])
        c = commentator_chain("成無己")
        self.assertEqual(c["chain_type"], "注家解釋鏈")
        # 成注被大量後世著作轉引（張卿子本以成注為底本）
        self.assertGreater(c["relay_hub"]["n_relaying_books"], 5)
        top_books = [t["book"] for t in c["relay_hub"]["top"]]
        self.assertIn("張卿子傷寒論", top_books)

    def test_text_trace_grounds_fragment(self):
        from hermes_shanghan.trace.chains import text_trace
        r = text_trace("观其脉证，知犯何逆，随证治之")   # 簡體輸入亦可回源
        self.assertTrue(r["matches"])
        self.assertEqual(r["matches"][0]["clause_id"], "SHL_SONGBEN_0016")

    def test_text_trace_honest_on_foreign_text(self):
        from hermes_shanghan.trace.chains import text_trace
        r = text_trace("春三月此謂發陳天地俱生萬物以榮")
        self.assertFalse(r.get("matches"))
        self.assertIn("無可回源", r.get("note", ""))


# ---------------------------------------------------------------------------
# 工具接線與治理
# ---------------------------------------------------------------------------
class TestTraceTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()
        from hermes_shanghan.agent.tools import get_registry
        cls.reg = get_registry()

    def test_trace_tool_stamped_mixed_with_sections(self):
        out = self.reg.call("shanghan_trace", {"query_type": "clause", "ref": "12"})
        self.assertEqual(out["tool"], "shanghan_trace")
        # 整體 mixed，逐節單獨標層（審核意見 1）
        self.assertEqual(out["evidence_level"], "mixed")
        self.assertIn("limitations", out)
        sections = out["trace"]["section_evidence_levels"]
        self.assertEqual(sections["clause"], "A 原文直述")
        self.assertEqual(sections["commentaries"], "C 注家解釋")
        self.assertIn("clause_id", json.dumps(out, ensure_ascii=False))

    def test_network_scope_separation(self):
        # 正文/輔助篇章分榜（審核意見 2）：默認 canonical 不得混入 AUX
        out = self.reg.call("shanghan_citation_network", {})
        self.assertEqual(out["scope"], "canonical")
        ids = [c["clause_id"] for c in out["top_cited_clauses"]]
        self.assertTrue(ids)
        self.assertFalse([i for i in ids if "AUX" in i])
        aux = self.reg.call("shanghan_citation_network", {"scope": "auxiliary"})
        aux_ids = [c["clause_id"] for c in aux["top_cited_clauses"]]
        self.assertTrue(aux_ids)
        self.assertTrue(all("AUX" in i for i in aux_ids))
        self.assertTrue(aux.get("ranking_note"))
        # 主路徑基於正文榜
        from hermes_shanghan.trace.builder import load_network
        net = load_network()
        self.assertFalse([m for m in net["main_paths"]
                          if "AUX" in m["clause_id"]])

    def test_network_tool_formula_target(self):
        out = self.reg.call("shanghan_citation_network", {"target": "桂枝湯"})
        self.assertEqual(out["target"]["kind"], "formula")
        self.assertGreater(out["target"]["total_mentions"], 100)

    def test_trace_tools_not_patient_exposed(self):
        # 方劑源流鏈含組成/劑量 → 患者模式不暴露（硬隔離）
        scoped = self.reg.for_role("patient")
        self.assertNotIn("shanghan_trace", scoped.names())
        out = scoped.call("shanghan_trace", {"query_type": "clause", "ref": "12"})
        self.assertIn("error", out)

    def test_modern_interface_honest_default(self):
        from hermes_shanghan.trace.modern import load_modern_trace
        r = load_modern_trace()
        self.assertFalse(r["available"])
        self.assertIn("不隨庫分發", r["note"])

    def test_network_scope_covers_all_fields(self):
        # 三輪評審方案 A：scope 貫穿時間切片/共引/突現/主路徑（審計器驗證）
        from hermes_shanghan.trace.scientometrics import audit_scope_consistency
        for scope in ("canonical", "auxiliary", "all"):
            payload = self.reg.call("shanghan_citation_network",
                                    {"scope": scope, "top_k": 20})
            report = audit_scope_consistency(payload, scope)
            self.assertTrue(report["ok"], report)

    def test_quote_check_misquotation(self):
        # A4 誤引檢測：評審給出的典型用例
        from hermes_shanghan.trace.chains import quote_check
        r = quote_check("营卫不和，桂枝汤主之")
        frag = {f["fragment"]: f for f in r["fragments"]}
        self.assertEqual(frag["營衛不和"]["verdict"], "後世歸納語（非原文）")
        self.assertIn("CLAIM_GZT_YINGWEI",
                      [c["claim_id"] for c in frag["營衛不和"]["related_claims"]])
        self.assertEqual(frag["桂枝湯主之"]["verdict"], "原文逐字")
        self.assertIn("SHL_SONGBEN_0012", frag["桂枝湯主之"]["verbatim_in"])
        self.assertIn("不能整句作為原文直引", r["verdict"])
        # 全原文輸入 → 可直引
        ok = quote_check("太陽之為病，脈浮，頭項強痛而惡寒")
        self.assertIn("可作原文直引", ok["verdict"])

    def test_audit_citation_reliability(self):
        # A2 引文邊審計
        from hermes_shanghan.schemas import read_jsonl
        from hermes_shanghan.trace.builder import _clause_texts
        from hermes_shanghan.trace.quotation import audit_citation
        cr = read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
        r = audit_citation("傷寒來蘇集", "SHL_SONGBEN_0012", _clause_texts(), cr)
        self.assertGreater(r["n_edges"], 0)
        for e in r["edges"]:
            self.assertIn(e["reliability"], ("高", "中", "低"))
        self.assertIn("error", audit_citation("不存在的書", "SHL_SONGBEN_0012",
                                              _clause_texts()))

    def test_goldset_roundtrip(self):
        # A3 金標準：抽樣→(以算法預測代人工)→評估 P/R/F1=1（機制自檢）
        import csv
        import tempfile
        from pathlib import Path

        from hermes_shanghan.trace.goldset import (CSV_FIELDS, build_sample,
                                                   evaluate)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gold.csv"
            s = build_sample(n=12, out_path=p)
            self.assertEqual(s["n_sampled"], 12)
            with p.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                row["human_clause_id"] = (row["algo_clause_id"]
                                          if row["algo_clause_id"] != "無" else "0")
                row["human_mode"] = row["algo_mode"]
            with p.open("w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            ev = evaluate(p)
            self.assertEqual(ev["clause_level"]["precision"], 1.0)
            self.assertEqual(ev["clause_level"]["recall"], 1.0)

    def test_claim_lineage_fields(self):
        # A5 觀點譜系：最早可見注家 + 術語首現
        from hermes_shanghan.trace.builder import load_claims
        gzt = next(c for c in load_claims()["claims"]
                   if c["claim_id"] == "CLAIM_GZT_YINGWEI")
        self.assertEqual(gzt["first_proponent"]["commentator"], "成無己")
        self.assertIn("在庫", gzt["first_proponent_note"])
        self.assertTrue(gzt["term_first_use"])
        self.assertTrue(gzt["interpretive_terms"])

    def test_herb_profile_and_formula_explain(self):
        # C10 藥解 / C11 方解
        from hermes_shanghan.apps.herbal import herb_profile
        from hermes_shanghan.trace.chains import formula_explain
        h = herb_profile("桂枝")
        self.assertGreater(h["n_formulas"], 30)
        self.assertEqual(h["top_partners"][0]["herb"], "甘草")
        self.assertIn("不編造", h["warnings"][0])
        f = formula_explain("桂枝湯")
        self.assertEqual(f["first_attestation"]["clause_id"], "SHL_SONGBEN_0012")
        self.assertTrue(f["differentials"])
        self.assertTrue(f["contraindications"] is not None)

    def test_coupling_scoped(self):
        # 四輪問題 2：文獻耦合逐 scope（著作條文集先過濾再算 Jaccard）
        from hermes_shanghan.trace.builder import load_network
        net = load_network()
        for scope in ("canonical", "auxiliary", "all"):
            self.assertIn("bibliographic_coupling", net["scoped"][scope])
        c = self.reg.call("shanghan_citation_network", {"scope": "canonical"})
        x = self.reg.call("shanghan_citation_network", {"scope": "auxiliary"})
        self.assertNotEqual(c["bibliographic_coupling"],
                            x["bibliographic_coupling"])

    def test_coupling_scope_semantics_synthetic(self):
        # 合成邊驗證耦合按域計算：兩書共享 12 條正文 + 12 條輔助
        from hermes_shanghan.trace.scientometrics import build_network
        edges = []
        for b in ("甲書", "乙書"):
            for i in range(1, 13):
                for cid in (f"SHL_SONGBEN_{i:04d}", f"SHL_SONGBEN_AUX_{i:04d}"):
                    edges.append({"target_kind": "clause", "clause_id": cid,
                                  "book_dir": b, "book": b, "author": "",
                                  "dynasty": "清", "layer": "C", "mode": "暗引",
                                  "coverage": 1.0, "longest_run": 20,
                                  "para_seq": i})
        net = build_network(edges, [])
        for scope in ("canonical", "auxiliary"):
            pair = net["scoped"][scope]["bibliographic_coupling"][0]
            self.assertEqual(pair["shared_clauses"], 12)
        self.assertEqual(net["scoped"]["all"]["bibliographic_coupling"][0]
                         ["shared_clauses"], 24)

    def test_audit_self_vs_relay_commentary(self):
        # 四輪問題 3：本書注文 ≠ 後世轉引
        from hermes_shanghan.schemas import read_jsonl
        from hermes_shanghan.trace.builder import _clause_texts
        from hermes_shanghan.trace.quotation import audit_citation
        cr = read_jsonl(config.RULES_COMMENTARY_DIR / "commentary_rules.jsonl")
        texts = _clause_texts()
        laisu = audit_citation("傷寒來蘇集", "SHL_SONGBEN_0012", texts, cr)
        laisu_comment_modes = {e["mode"] for e in laisu["edges"]
                               if "注文" in e["mode"]}
        for m in laisu_comment_modes:
            self.assertIn("self_commentary", m)   # 柯琴書中命中柯琴注=本書注文
        zqz = audit_citation("張卿子傷寒論", "SHL_SONGBEN_0012", texts, cr)
        relay = [e for e in zqz["edges"] if "relay_commentary" in e["mode"]]
        self.assertTrue(relay)                     # 張卿子本轉引成無己注
        self.assertTrue(any("成無己" in f for e in relay for f in e["flags"]))

    def test_herb_and_explain_registered_as_tools(self):
        # 四輪問題 4：藥解/方解成為完整註冊表工具（自動導出 MCP/OpenAI 規格）
        names = self.reg.names()
        self.assertIn("shanghan_herb_profile", names)
        self.assertIn("shanghan_formula_explain", names)
        h = self.reg.call("shanghan_herb_profile", {"herb": "桂枝"})
        # 五輪評審：A 層材料的派生統計標 A-derived，不冒充原文直述
        self.assertEqual(h["evidence_level"], "A-derived")
        f = self.reg.call("shanghan_formula_explain", {"formula": "桂枝湯"})
        self.assertEqual(f["evidence_level"], "mixed")
        # 患者模式不暴露（含劑量）
        scoped = self.reg.for_role("patient")
        self.assertNotIn("shanghan_herb_profile", scoped.names())
        self.assertNotIn("shanghan_formula_explain", scoped.names())

    def test_symptom_layers_three_tier(self):
        # 四輪問題 5：首見/全書聚合/特殊上下文三層口徑
        from hermes_shanghan.trace.chains import formula_explain
        f = formula_explain("桂枝湯")
        layers = f["symptom_layers"]
        self.assertEqual(layers["first_attestation"]["clause_id"],
                         "SHL_SONGBEN_0012")
        self.assertIn("嗇嗇惡寒", layers["first_attestation"]["symptoms"])
        self.assertTrue(layers["aggregate_all_clauses"])
        ctx15 = next(s for s in layers["special_context"]
                     if s["clause_id"] == "SHL_SONGBEN_0015")
        self.assertIn("誤治", ctx15["context"])
        self.assertIn("不得徑作標準方證核心證", layers["note"])

    def test_goldset_stratified(self):
        # 四輪問題 6：分層抽樣（朝代×預測模式，零隨機）
        from hermes_shanghan.trace.goldset import build_sample
        s = build_sample(n=16, stratify=True)
        self.assertGreaterEqual(s["n_strata"], 3)
        strata = {r["stratum"] for r in s["rows"]}
        self.assertTrue(any("×無" in st for st in strata))   # 含負例層
        # 可復現：重跑一致
        s2 = build_sample(n=16, stratify=True)
        self.assertEqual(s["rows"], s2["rows"])
        # 七輪修復：請求 n 恰返回 n（層多於 n 時取最大 n 層，不超額）
        self.assertEqual(len(s["rows"]), 16)
        self.assertEqual(len(build_sample(n=6, stratify=True)["rows"]), 6)

    def test_gold_eval_rows_web_roundtrip(self):
        # human-in-the-loop 標註閉環的 Web 路徑（rows 進 rows 出，不落盤）
        from hermes_shanghan.server.service import get_service
        svc = get_service()
        gs = svc.gold_sample(n=6)
        rows = gs["rows"]
        for r in rows:
            r["human_clause_id"] = (r["algo_clause_id"]
                                    if r["algo_clause_id"] != "無" else "0")
            r["human_mode"] = r["algo_mode"]
        ev = svc.gold_eval(rows)
        self.assertEqual(ev["clause_level"]["f1"], 1.0)

    def test_api_tools_lists_full_registry(self):
        from hermes_shanghan.server.service import get_service
        t = get_service().tools()
        self.assertEqual(len(t["tools"]), len(self.reg.names()))

    def test_herb_role_evidence_and_bencao_honesty(self):
        # 七輪：方中作用證據（量變致新方事件）+ 性味提不到不編造
        from hermes_shanghan.apps.herbal import herb_profile
        h = herb_profile("桂枝")
        events = {(r["base"], r["modified"], r["event"])
                  for r in h["role_evidence"]}
        self.assertIn(("桂枝湯", "桂枝加桂湯", "劑量調整"), events)
        for e in h["bencao_layer"].get("excerpts", []):
            if "nature_flavor" in e:   # 出現時必為逐字提取且帶層標
                self.assertIn("本草層", e["nature_flavor"]["source_layer"])

    def test_differential_routing_simplified_variants(self):
        # 七輪：鑒別路由穩定命中（簡體/口語變體）
        from hermes_shanghan.agent.agent import ShanghanAgent
        for q in ("桂枝汤和麻黄汤怎么区分？", "桂枝湯與麻黃湯有什麼差別",
                  "比较一下桂枝汤和麻黄汤"):
            out = ShanghanAgent().ask(q, role="doctor")
            self.assertIn("shanghan_differential", out["tools_used"], q)

    def test_tools_workbench_static(self):
        root = config.REPO_ROOT / "hermes_shanghan" / "server" / "static"
        js = (root / "app.js").read_text(encoding="utf-8")
        html = (root / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-view="tools"', html)
        self.assertIn("views.tools", js)
        for feature in ("/api/tools", "/api/gold-sample", "/api/gold-eval",
                        "shanghan_library", "deep-research",
                        "shanghan_eval_metrics"):
            self.assertIn(feature, js)

    def test_term_chain(self):
        # 術語譜系：營衛不和非原文、在庫首現注家可查
        from hermes_shanghan.trace.chains import term_chain
        t = term_chain("營衛不和")
        self.assertEqual(t["verbatim_in_original"], [])
        self.assertIn("後世術語", t["evidence_grade"])
        self.assertTrue(t["commentarial_chronology"])
        self.assertTrue(t["related_claims"])
        # 原文逐字術語（胃家實，第180條提綱）
        t2 = term_chain("胃家實")
        self.assertIn("SHL_SONGBEN_0180", t2["verbatim_in_original"])
        self.assertIn("原文逐字", t2["evidence_grade"])

    def test_dispute_chain_structured_no_verdict(self):
        # 注家爭議結構化：呈現證據結構，不裁決對錯
        from hermes_shanghan.trace.chains import dispute_chain
        d = dispute_chain("12")
        self.assertEqual(d["chain_type"], "注家爭議結構化")
        self.assertGreaterEqual(d["n_commentators"], 5)
        v0 = d["views"][0]
        for key in ("closeness_to_original", "posthoc_terms", "analytic_focus",
                    "school", "dynasty"):
            self.assertIn(key, v0)
        self.assertIn("不可裁決", d["undecidable_note"])
        self.assertIn("E 啟發式", d["section_evidence_levels"]
                      ["divergence_types_present"])

    def test_compare_chain(self):
        from hermes_shanghan.trace.chains import compare_chain
        c = compare_chain("柯琴 vs 尤怡")
        self.assertEqual(c["a"]["commentator"], "柯琴")
        self.assertEqual(c["b"]["commentator"], "尤怡")
        self.assertTrue(c["agreement"])
        self.assertTrue(c["top_divergent_clauses"])
        self.assertIn("error", compare_chain("柯琴"))   # 格式校驗

    def test_formula_aliases_merged_but_separate(self):
        # 異名歸並：陽旦湯計量與正名分列；歧義名標不可合併
        from hermes_shanghan.trace.builder import load_formula_mentions
        from hermes_shanghan.trace.chains import formula_chain
        r = formula_chain("桂枝湯")
        aliases = r["name_transmission"]["aliases"]
        yd = next(a for a in aliases if a["alias"] == "陽旦湯")
        self.assertTrue(yd["same_formula"])
        self.assertGreater(yd["alias_mentions"], 10)
        # 陽旦湯在計量資產中獨立成行（不混入桂枝湯計數）
        names = {f["formula"] for f in load_formula_mentions()["formulas"]}
        self.assertIn("陽旦湯", names)
        xch = formula_chain("小柴胡湯")["name_transmission"]["aliases"]
        self.assertFalse(next(a for a in xch
                              if a["alias"] == "柴胡湯")["same_formula"])

    def test_bencao_layer_gated_and_labeled(self):
        from hermes_shanghan.apps.herbal import bencao_evidence, herb_profile
        h = herb_profile("桂枝")
        self.assertIn("bencao_layer", h)
        self.assertIn("旁證", h["section_evidence_levels"]["bencao_layer"])
        from hermes_shanghan.corpus import library
        if library.is_available():
            bc = bencao_evidence("桂枝")
            self.assertTrue(bc["available"])
            self.assertTrue(bc["excerpts"])
            self.assertIn("旁證", bc["note"])
        else:
            self.assertFalse(bencao_evidence("桂枝")["available"])

    def test_webui_static_has_new_views_and_attribution(self):
        # 前端：新模塊在冊 + 移動端媒體查詢 + 研發來源標識
        root = config.REPO_ROOT / "hermes_shanghan" / "server" / "static"
        html = (root / "index.html").read_text(encoding="utf-8")
        js = (root / "app.js").read_text(encoding="utf-8")
        css = (root / "app.css").read_text(encoding="utf-8")
        self.assertIn("醫哲未來人工智能研究院", html)
        for view in ("trace", "herbs", "bianzheng"):
            self.assertIn(f'data-view="{view}"', html)
            self.assertIn(f"views.{view}", js)
        self.assertIn("@media (max-width: 820px)", css)
        self.assertIn("/api/trace", js)
        self.assertIn("shanghan_intake", js)

    def test_scan_library_with_fixture(self):
        # 全庫掃描（引用方=任意醫籍）：合成最小庫驗證端到端，不依賴真實下載
        import tempfile
        from pathlib import Path

        from hermes_shanghan.trace.builder import _clause_texts
        from hermes_shanghan.trace.quotation import scan_library
        texts = _clause_texts()
        clause1 = texts["SHL_SONGBEN_0001"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            book = root / "books" / "測試醫籍"
            book.mkdir(parents=True)
            (book / "index.txt").write_text(
                "======測試醫籍======\n\n<book>\n書名=測試醫籍\n作者=測者\n"
                "朝代=清\n分類=傷寒\n</book>\n\n=====卷一=====\n\n"
                f"仲景曰：{clause1}。此論太陽之綱領也。\n",
                encoding="utf-8")
            (root / "catalog.json").write_text(json.dumps({
                "units": [{"id": "測試醫籍", "title": "測試醫籍", "author": "測者",
                           "dynasty": "清", "category": "傷寒",
                           "files": ["index.txt"], "parent": "", "sub_books": []}],
            }, ensure_ascii=False), encoding="utf-8")
            res = scan_library(texts, root=root)
            self.assertTrue(res["available"])
            hit = next(e for e in res["edges"]
                       if e["clause_id"] == "SHL_SONGBEN_0001")
            self.assertEqual(hit["mode"], "明引")
            self.assertEqual(hit["layer"], "旁證")
            self.assertEqual(hit["book"], "測試醫籍")
        with tempfile.TemporaryDirectory() as td:
            res = scan_library(texts, root=Path(td))
            self.assertFalse(res["available"])   # 未下載時如實返回，不觸發下載

    def test_research_loop_covers_citation_dimension(self):
        from hermes_shanghan.agent.research_loop import DeepResearcher
        d = DeepResearcher(max_rounds=3).run("桂枝湯的歷代引用與傳播")
        self.assertEqual(d["uncovered_dimensions"], [])
        self.assertGreaterEqual(d["coverage"]["引文傳播"], 1)
        f = next(f for f in d["findings"] if f["dimension"] == "引文傳播")
        self.assertTrue(f["citation_ok"])


if __name__ == "__main__":
    unittest.main()
