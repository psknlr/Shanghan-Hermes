"""Paper augmentation layer + LLM performance-unlock tests.

Covers, without any network:
  * offline paper drafting (local backend) — 計量結果解讀 woven from real
    research numbers, citation-verified footer, type-specific results;
  * scripted "real-model" drafting — fabricated clause_ids are flagged;
  * task-tiered max_tokens, full-text synthesize evidence, task-call caching,
    and poe/minimax OpenAI-compatible routing.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_shanghan import config
from hermes_shanghan.llm.client import LLMClient
from hermes_shanghan.llm.config import LLMSettings
from hermes_shanghan.llm.providers import LiteLLMProvider, ScriptedProvider


def _ensure_artifacts():
    if not (config.RULES_INITIAL_DIR / "initial_rules.jsonl").exists():
        from hermes_shanghan.orchestrator import run_pipeline
        run_pipeline(verbose=False)


def _writer(llm_client=None):
    from hermes_shanghan.orchestrator import Artifacts
    from hermes_shanghan.paper.writer import PaperWriter
    art = Artifacts()
    return PaperWriter(art.clauses, art.initial_rules, art.formula_rules,
                       art.six_channel_rules, art.mistreatment_rules,
                       art.differential_rules,
                       commentary_rules=art.commentary_rules,
                       llm_client=llm_client)


class TestPaperAugmentationOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def _generate(self, paper_type, topic="", **kw):
        w = _writer()
        with tempfile.TemporaryDirectory() as tmp:
            path = w.generate(paper_type=paper_type, topic=topic,
                              out_dir=Path(tmp), **kw)
            text = path.read_text(encoding="utf-8")
            meta = json.loads((path.parent / "paper_meta.json")
                              .read_text(encoding="utf-8"))
        return text, meta

    def test_quant_interpretation_from_research_assets(self):
        text, meta = self._generate("formula_pattern", topic="桂枝湯類方證")
        self.assertIn("### 6.9 計量結果增益層解讀", text)
        # real numbers from the mined data, not placeholders
        self.assertIn("桂枝湯", text)
        self.assertIn("共現", text)
        # machine prose passed the citation guard and cites verifiable clauses
        self.assertIn("【增益層引用核驗】", text)
        self.assertIn("已核實條文：SHL_SONGBEN_", text)
        self.assertNotIn("未能核實的條文編號", text)
        self.assertEqual(meta["llm_backend"], "local")
        self.assertIn("quant_interpretation", meta["llm_sections"])
        self.assertTrue(meta["citation_report"]["ok"])

    def test_no_llm_flag_skips_augmentation(self):
        text, meta = self._generate("formula_pattern", use_llm=False)
        self.assertNotIn("計量結果增益層解讀", text)
        self.assertEqual(meta["llm_backend"], "disabled")
        # deterministic skeleton still complete（十九輪：新增敘述層章節）
        self.assertIn("## 5 方證各論", text)
        self.assertIn("## 6 計量結果分述", text)
        self.assertIn("## 8 討論", text)
        self.assertIn("## 9 結論", text)

    def test_mistreatment_table_no_longer_leaks_into_all_types(self):
        text, _ = self._generate("formula_pattern")
        self.assertNotIn("誤治傳變路徑（節選）", text)
        text, _ = self._generate("mistreatment")
        self.assertIn("誤治傳變路徑（節選）", text)

    def test_network_pharmacology_specific_sections(self):
        text, _ = self._generate("network_pharmacology")
        self.assertIn("方-證共現網絡", text)
        self.assertIn("關聯證候數", text)
        self.assertIn("高頻藥物", text)

    def test_commentary_compare_specific_sections(self):
        text, _ = self._generate("commentary_compare", topic="桂枝湯歷代注釋")
        self.assertIn("多注家對齊示例", text)
        self.assertIn("C層", text)

    def test_methodology_specific_sections(self):
        text, _ = self._generate("methodology")
        self.assertIn("審核閘門通過情況", text)
        self.assertIn("| schema |", text)
        self.assertIn("| evidence |", text)


class TestPaperScriptedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_artifacts()

    def test_fabricated_clause_in_draft_is_flagged(self):
        draft = {"introduction": "依 SHL_SONGBEN_0012 立論。",
                 "quant_interpretation": "而 SHL_SONGBEN_9999 並不存在。",
                 "discussion": "略。", "conclusion": "略。"}
        client = LLMClient(provider=ScriptedProvider([json.dumps(
            draft, ensure_ascii=False)]))
        w = _writer(llm_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            path = w.generate(paper_type="formula_pattern", out_dir=Path(tmp))
            text = path.read_text(encoding="utf-8")
            meta = json.loads((path.parent / "paper_meta.json")
                              .read_text(encoding="utf-8"))
        self.assertIn("依 SHL_SONGBEN_0012 立論", text)
        self.assertIn("未能核實的條文編號（請勿採信）：SHL_SONGBEN_9999", text)
        self.assertIn("SHL_SONGBEN_9999", meta["citation_report"]["unsupported"])
        self.assertIn("SHL_SONGBEN_0012", meta["citation_report"]["verified"])

    def test_empty_model_output_falls_back_to_templates(self):
        client = LLMClient(provider=ScriptedProvider(["not json at all"]))
        w = _writer(llm_client=client)
        with tempfile.TemporaryDirectory() as tmp:
            path = w.generate(paper_type="formula_pattern", out_dir=Path(tmp))
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("計量結果增益層解讀", text)
        self.assertIn("## 8 討論", text)  # template fallback intact


class TestModelPerformanceUnlocks(unittest.TestCase):
    def test_max_tokens_tiered_by_task(self):
        s = LLMSettings()
        self.assertGreaterEqual(s.max_tokens_for("paper"), 8192)
        self.assertGreaterEqual(s.max_tokens_for("synthesize"), 4096)
        self.assertEqual(s.max_tokens_for(None), s.max_tokens)
        # an explicit higher user setting always wins
        s2 = LLMSettings(max_tokens=20000)
        self.assertEqual(s2.max_tokens_for("paper"), 20000)

    def test_synthesize_passes_full_evidence_text(self):
        long_text = "往來寒熱，胸脅苦滿，嘿嘿不欲飲食，心煩喜嘔" * 10  # 200 chars
        provider = ScriptedProvider(["答"])
        client = LLMClient(provider=provider)
        client.synthesize("問", [{"clause_id": "SHL_SONGBEN_0096",
                                   "text": long_text}])
        user_msg = provider.calls[0]["messages"][-1]["content"]
        self.assertIn(long_text[:200], user_msg)  # not truncated at 80 chars

    def test_task_calls_are_cached(self):
        provider = ScriptedProvider(['{"verdict": "pass"}'])
        client = LLMClient(provider=provider)
        client._backend = "litellm"  # caching only arms on real backends
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"}]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("hermes_shanghan.llm.cache._cache_dir",
                            return_value=Path(tmp)):
                r1 = client.chat(msgs, task="critic", json_mode=True)
                r2 = client.chat(msgs, task="critic", json_mode=True)
        self.assertEqual(r1.content, r2.content)
        self.assertEqual(len(provider.calls), 1)          # second hit the cache
        self.assertEqual(client.usage["cache_hits"], 1)

    def test_cache_key_separates_tasks(self):
        from hermes_shanghan.llm import cache as cache_mod
        msgs = [{"role": "user", "content": "x"}]
        k1 = cache_mod.cache_key("m", msgs, None, 0.0, task="critic")
        k2 = cache_mod.cache_key("m", msgs, None, 0.0, task="extract_rule")
        self.assertNotEqual(k1, k2)


class TestOpenAICompatibleRouting(unittest.TestCase):
    def test_poe_route(self):
        with mock.patch.dict("os.environ", {"POE_API_KEY": "pk"}, clear=False):
            model, kw = LiteLLMProvider._resolve_route(
                LLMSettings(model="poe/Claude-Sonnet-4.5"))
        self.assertEqual(model, "openai/Claude-Sonnet-4.5")
        self.assertEqual(kw["api_base"], "https://api.poe.com/v1")
        self.assertEqual(kw["api_key"], "pk")

    def test_minimax_route_with_base_override(self):
        env = {"MINIMAX_API_KEY": "mk",
               "MINIMAX_API_BASE": "https://api.minimaxi.com/v1"}
        with mock.patch.dict("os.environ", env, clear=False):
            model, kw = LiteLLMProvider._resolve_route(
                LLMSettings(model="minimax/MiniMax-M2"))
        self.assertEqual(model, "openai/MiniMax-M2")
        self.assertEqual(kw["api_base"], "https://api.minimaxi.com/v1")
        self.assertEqual(kw["api_key"], "mk")

    def test_native_prefixes_untouched(self):
        for m in ("anthropic/claude-opus-4-8", "azure/my-deployment",
                  "openai/gpt-4.1"):
            model, kw = LiteLLMProvider._resolve_route(LLMSettings(model=m))
            self.assertEqual(model, m)
            self.assertEqual(kw, {})


if __name__ == "__main__":
    unittest.main()
