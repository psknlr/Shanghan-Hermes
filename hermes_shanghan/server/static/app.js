"use strict";
// ---------- tiny helpers ----------
const $ = (s, r = document) => r.querySelector(s);
// 認證（十二輪 P0：啟用 HERMES_API_KEYS/SERVER_TOKEN 後 UI 仍可用）：
// token 存 localStorage('hermes_token')，所有請求帶 Authorization 頭；
// 401 時提示設置（新運行中心 /console.html 有 token 輸入框，同一存儲鍵）
function authHeaders() {
  const t = localStorage.getItem("hermes_token");
  return t ? { "Authorization": "Bearer " + t } : {};
}
const api = {
  async get(p) {
    const r = await fetch(p, { headers: authHeaders() });
    if (r.status === 401) throw new Error("401 未授權：請在 /console.html 設置訪問 token");
    return r.json();
  },
  async post(p, b) {
    const r = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify(b || {}) });
    if (r.status === 401) throw new Error("401 未授權：請在 /console.html 設置訪問 token");
    return r.json();
  },
};
function el(tag, attrs, children) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === "class") e.className = attrs[k];
    else if (k === "html") e.innerHTML = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  }
  (Array.isArray(children) ? children : children != null ? [children] : []).forEach(c => {
    if (c == null) return;
    e.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  });
  return e;
}
const esc = s => (s == null ? "" : String(s)).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const layerLetter = s => { const m = /([A-E])/.exec(s || ""); return m ? m[1] : "A"; };

// ---------- i18n（十八輪：繁/简/EN 界面 + 條文簡繁顯示切換） ----------
// 简体模式＝顯示層字級自動轉換（/api/charmap 領域字表；原文以繁體為準）；
// EN 模式＝界面骨架詞典翻譯，古籍內容保持中文。切換即持久化並重載。
let LANG = localStorage.getItem("hermes_lang") || "zh-Hant";
let T2S_MAP = null;
const I18N_EN = {
  "總覽": "Overview", "智能體": "Agents", "原文檢索": "Text Search",
  "方證匹配": "Pattern Match", "方證鑒別": "Differentiation",
  "六經教學": "Six Channels", "誤治傳變": "Mistreatment Paths",
  "科研挖掘": "Research Mining", "溯源工作台": "Provenance Bench",
  "方藥檔案": "Herbs & Formulas", "辨證閉環": "Diagnosis Loop",
  "工具台": "Tool Bench", "論文生成": "Paper Writer", "Skill 庫": "Skill Library",
  "接入/關於": "About & Integrations", "運行中心（新）": "Run Center",
  "全量古籍工作台": "Classics Bench",
};
function t2sText(s) { if (!T2S_MAP) return s; let out = ""; for (const ch of s) out += T2S_MAP[ch] || ch; return out; }
function convertTree(root) {
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let n; while ((n = w.nextNode())) { const v = n.nodeValue, c = t2sText(v); if (c !== v) n.nodeValue = c; }
}
async function applyLang() {
  document.documentElement.lang = LANG;
  const sel = $("#lang-select");
  if (sel) { sel.value = LANG; sel.addEventListener("change", () => { localStorage.setItem("hermes_lang", sel.value); location.reload(); }); }
  document.querySelectorAll("#nav button span").forEach(sp => {
    if (LANG === "en") sp.textContent = I18N_EN[sp.textContent] || sp.textContent;
  });
  if (LANG === "zh-Hans") {
    try { T2S_MAP = (await api.get("/api/charmap")).t2s || {}; } catch (e) { T2S_MAP = {}; }
    convertTree(document.body);
    // 動態內容（視圖切換/載入更多/抽屜）一律經觀察器轉換（冪等）
    new MutationObserver(muts => muts.forEach(m => (m.addedNodes || []).forEach(nd => {
      if (nd.nodeType === 3) { const c = t2sText(nd.nodeValue); if (c !== nd.nodeValue) nd.nodeValue = c; }
      else if (nd.nodeType === 1) convertTree(nd);
    }))).observe(document.body, { childList: true, subtree: true });
  }
}

// ---------- clause drawer（十八輪：返回棧導航） ----------
const drawerHistory = [];
let currentClauseRef = null;
function updateBackBtn() { const b = $("#drawer-back"); if (b) b.style.display = drawerHistory.length ? "" : "none"; }
async function openClause(ref, push = true) {
  const d = $("#drawer"), sc = $("#drawer-scrim"), body = $("#drawer-body");
  d.classList.add("open"); sc.classList.add("open");
  if (push && currentClauseRef && String(currentClauseRef) !== String(ref)) drawerHistory.push(currentClauseRef);
  currentClauseRef = ref;
  updateBackBtn();
  $("#drawer-title").textContent = "條文 " + ref;
  body.innerHTML = '<div class="loading">載入中…</div>';
  const c = await api.get("/api/clause/" + encodeURIComponent(ref));
  if (c.error) { body.innerHTML = '<p class="muted">' + esc(c.error) + "</p>"; return; }
  body.innerHTML = "";
  $("#drawer-title").textContent = c.clause_id + (c.clause_number ? "（第" + c.clause_number + "條）" : "");
  body.append(
    el("div", { class: "row small muted" }, [c.chapter + " · " + (c.six_channel || ""), el("span", { class: "layer " + layerLetter(c.layer_label) }, layerLetter(c.layer_label))]),
    el("p", { class: "classical" }, c.text),
  );
  const ent = c.entities || {};
  const entRow = (k, arr) => arr && arr.length ? el("div", { class: "kv" }, [el("b", {}, k), el("span", {}, arr.join("、"))]) : null;
  body.append(el("div", { class: "card" }, [
    el("div", { class: "section-title" }, "實體標註"),
    entRow("症狀", ent.symptoms), entRow("否定", ent.negated_findings), entRow("脈象", ent.pulse),
    entRow("方劑", ent.formulas), entRow("治法", ent.therapy), entRow("禁忌", ent.contraindications),
    entRow("誤治", ent.mistreatment), entRow("預後", ent.prognosis),
  ]));
  if (c.formula_blocks && c.formula_blocks.length) {
    const fb = c.formula_blocks.map(b => el("div", {}, [
      el("b", { class: "section-title" }, b.formula_name || "方"),
      el("div", { class: "small" }, (b.composition || []).map(x => x.herb + (x.dose_processing ? "（" + x.dose_processing + "）" : "")).join("　")),
      b.administration ? el("div", { class: "small muted" }, "服法：" + b.administration) : null,
    ]));
    body.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "方劑塊"), ...fb]));
  }
  if (c.initial_rules && c.initial_rules.length) {
    body.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "抽取規則 " + c.initial_rules.length + " 條"),
      ...c.initial_rules.map(r => el("div", { class: "kv small" }, [el("b", {}, r.type.replace("_rule", "")), el("span", {}, (r.strength ? "【" + r.strength + "】" : "")), el("span", { class: "pill" }, r.release)]))]));
  }
  if (c.relations && c.relations.length) {
    body.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "條文關係（目標皆可點閱）"),
      ...c.relations.slice(0, 8).map(r => {
        const row = el("div", { class: "small", style: "padding:3px 0" }, [
          el("span", { class: "pill" }, r.relation_type), " ",
          r.clause_id && r.clause_id.startsWith("SHL") ? clauseChip(r.clause_id) : sourceRefChip(r.clause_id), " ",
          el("span", { class: "muted" }, (r.description || "").slice(0, 40))]);
        return row;
      })]));
  }
  if (c.variants && c.variants.length) {
    body.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, [el("span", { class: "layer B" }, "B"), " 版本異文"]),
      ...c.variants.map(v => el("div", { class: "evi" }, [el("div", { class: "ct" }, v.text), el("div", { class: "meta" }, v.book + " · 相似度 " + v.similarity + (v.differences && v.differences.length ? " · " + v.differences.join("；") : ""))]))]));
  }
  if (c.commentaries && c.commentaries.length) {
    // 智能化注家層：貼近原文度/學派/取徑逐家標注（commentary_analysis）
    const ca = c.commentary_analysis || {};
    const meta = {};
    (ca.views || []).forEach(v => meta[v.commentator] = v);
    body.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, [el("span", { class: "layer C" }, "C"), " 注家解釋",
        ca.divergence_types_present && ca.divergence_types_present.length ? el("span", { class: "small muted" }, "　分歧取徑：" + ca.divergence_types_present.join("、")) : null]),
      ...c.commentaries.map(v => {
        const m = meta[v.commentator] || {};
        return el("div", { class: "evi" }, [
          el("div", { class: "ct", style: "font-size:13px" }, v.text),
          el("div", { class: "meta" }, [
            el("b", {}, v.commentator), m.dynasty ? "（" + m.dynasty + "）" : "",
            v.book ? el("span", {}, "《" + v.book + "》" + (v.chapter ? "·" + v.chapter : "")) : null,
            m.school ? el("span", { class: "pill" }, m.school) : null,
            m.closeness_to_original != null ? el("span", { class: "pill" }, "貼近原文 " + m.closeness_to_original) : null,
            (m.analytic_focus || []).length ? el("span", { class: "pill" }, "取徑：" + m.analytic_focus.join("/")) : null,
            (m.posthoc_terms || []).length ? el("span", { class: "small muted" }, "　後世術語：" + m.posthoc_terms.join("、")) : null,
          ])]);
      }),
      ca.note ? el("p", { class: "small muted" }, ca.note) : null]));
  }
  const hc = c.historical_citations;
  if (hc && hc.n_books) {
    const wrap = el("div", { class: "card" }, [el("div", { class: "section-title" },
      "歷代古籍相關條目（" + hc.n_books + " 部書 · " + hc.n_edges + " 段引用，逐字回源）")]);
    (hc.by_dynasty || []).forEach(d => {
      const det = el("details", { style: "margin:4px 0", open: d === hc.by_dynasty[0] ? "" : null }, [
        el("summary", {}, [el("b", {}, d.dynasty), el("span", { class: "muted small" }, "　" + d.books.length + " 部書")]),
        ...d.books.map(b => el("div", { style: "padding:4px 0 4px 10px" }, [
          el("div", { class: "small" }, [el("b", {}, "《" + b.book + "》"), el("span", { class: "muted" }, (b.author ? b.author + "　" : "") + b.n_citing_paragraphs + " 段引用")]),
          ...(b.passages || []).map(p => el("div", { class: "evi" }, [
            el("div", { class: "ct", style: "font-size:12px" }, p.excerpt || p.matched_span),
            el("div", { class: "meta" }, [el("span", { class: "pill" }, p.mode), " §" + p.chapter + " · 覆蓋 " + p.coverage,
              p.via_commentator ? el("span", { class: "muted" }, "　經" + p.via_commentator + "注轉引") : null])]))]))]);
      wrap.append(det);
    });
    wrap.append(el("p", { class: "small muted" }, hc.note || ""));
    body.append(wrap);
  }
  body.append(clauseAiCard(c));
}
// AI 解讀 + 圍繞條文的智能體對話（十八輪）：走 /api/agent，模型自主調用
// 工具取證，回答經引用核驗；local 後端同樣可用（確定性取證）。
function clauseAiCard(c) {
  const label = c.clause_id + (c.clause_number ? "（第" + c.clause_number + "條）" : "");
  const out = el("div", {});
  const inp = el("input", { type: "text", placeholder: "圍繞本條提問，如：此條与桂枝湯證如何鑒別？" });
  async function ask(q) {
    out.prepend(el("div", { class: "loading" }, "智能體取證中…"));
    let r;
    try { r = await api.post("/api/agent", { question: q, role: "student", max_steps: 5 }); }
    catch (e) { r = { answer: "請求失敗：" + e.message }; }
    out.querySelector(".loading")?.remove();
    const card = el("div", { class: "evi", style: "margin-top:6px" }, [
      el("div", { class: "small muted" }, "問：" + q.slice(0, 60)),
      el("div", { class: "answer-text", style: "font-size:13px" }, r.answer || "(無回答)")]);
    const cr = r.citation_report;
    if (cr) card.append(el("div", { class: "cite-banner " + (cr.ok && cr.has_any_citation ? "cite-ok" : "cite-warn") },
      cr.ok && cr.has_any_citation ? "✓ 引用已核驗：" + (cr.verified || []).join("、")
        : "⚠ " + ((cr.unsupported || []).length ? "未核實：" + cr.unsupported.join("、") : "無可核驗條文編號")));
    if ((r.tools_used || []).length) card.append(el("div", { class: "small muted" }, "工具：" + r.tools_used.join("、")));
    if ((r.evidence_clause_ids || []).length) card.append(clauseChips(r.evidence_clause_ids.slice(0, 8)));
    out.prepend(card);
  }
  inp.addEventListener("keydown", e => { if (e.key === "Enter") { const q = inp.value.trim(); if (q) { ask("圍繞《傷寒論》條文 " + label + " 回答：" + q); inp.value = ""; } } });
  return el("div", { class: "card" }, [
    el("div", { class: "section-title" }, "🤖 AI 解讀與對話（智能體自動取證 · 引用核驗）"),
    el("div", { class: "row" }, [
      el("button", { class: "btn sm", onclick: () => ask("請解讀 " + label + "：原文含義、辨證要點、所涉方證與相關鑒別。") }, "AI 解讀本條"),
      el("div", { style: "flex:1" }, inp)]),
    out,
    el("p", { class: "small muted" }, "回答中的條文編號逐一過 CitationGuard；接入真實大模型後解讀更完整。")]);
}
// 非條文關係目標（異文「書:章節」/ 注文「書:pN」）→ 點閱原始段落
function sourceRefChip(target) {
  const i = (target || "").indexOf(":");
  if (i < 0) return el("span", { class: "muted" }, target);
  const book = target.slice(0, i), ref = target.slice(i + 1);
  const bodyEl = el("div", { class: "lib-passage", style: "display:none" });
  let loaded = false;
  const chip = el("span", { class: "clause-chip", onclick: async () => {
    if (bodyEl.style.display === "none") {
      bodyEl.style.display = "";
      if (!loaded) {
        bodyEl.innerHTML = '<div class="loading">讀取原文…</div>';
        try {
          const r = await api.post("/api/source/passage", { book, ref });
          bodyEl.innerHTML = "";
          if (r.error) bodyEl.append(el("p", { class: "muted small" }, r.error + ((r.available_chapters || []).length ? "　章節：" + r.available_chapters.slice(0, 6).join("、") : "")));
          else {
            bodyEl.append(el("div", { class: "small muted" }, "《" + r.book + "》·" + r.chapter));
            (r.paragraphs || []).forEach(p => bodyEl.append(el("div", { class: "evi" }, [el("div", { class: "ct", style: "font-size:12.5px" }, p.text), el("div", { class: "meta" }, "第 " + p.para_seq + " 段")])));
            if (r.note) bodyEl.append(el("p", { class: "small muted" }, r.note));
          }
          loaded = true;
        } catch (e) { bodyEl.innerHTML = '<p class="muted small">讀取失敗：' + esc(e.message) + "</p>"; }
      }
    } else bodyEl.style.display = "none";
  } }, ["▸ ", target]);
  return el("span", {}, [chip, bodyEl]);
}
function closeDrawer() { $("#drawer").classList.remove("open"); $("#drawer-scrim").classList.remove("open"); drawerHistory.length = 0; currentClauseRef = null; updateBackBtn(); }
function clauseChip(id) { return el("span", { class: "clause-chip", onclick: () => openClause(id) }, ["⌖ ", id]); }
function clauseChips(ids) { const w = el("div", {}); (ids || []).forEach(id => w.appendChild(clauseChip(id))); return w; }

// ---------- views ----------
const views = {};
let META = {};

views.dashboard = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("總覽", "《傷寒論》知識底座一覽 · 確定性規則 + 可選 LLM 增益"));
  const s = await api.get("/api/stats");
  const cards = [
    ["條文", s.canonical, "宋本編號條文"], ["初始規則", s.initial_rules, "17 類 · 逐條回源"],
    ["方證規則", s.formula_pattern_rules, "核心證/組成/加減"], ["六經規則", s.six_channel_rules, "提綱/亞型/主方"],
    ["誤治路徑", s.mistreatment_rules, "誤治→變證→救治"], ["鑒別規則", s.differential_rules, "多軸對比"],
    ["合併規則", s.merged_rules, "引用+證據鏈"], ["Skill", s.skills, "可調用技能"],
    ["條文關係", s.clause_relations, "7 類關係邊"], ["異文", s.variant_rules, "B 層"],
    ["注釋", s.commentary_rules, "C 層"], ["審計記錄", s.audits, "6 道閘門"],
  ];
  main.append(el("div", { class: "grid cols-4" }, cards.map(([l, n, d]) =>
    el("div", { class: "card statbox" }, [el("div", { class: "num" }, (n ?? 0).toLocaleString()), el("div", { class: "lbl" }, l), el("div", { class: "small muted" }, d)]))));

  const lv = s.release_levels || {};
  const tot = (lv.gold || 0) + (lv.silver || 0) + (lv.bronze || 0) || 1;
  const barRow = (k, cls) => el("div", { style: "margin:8px 0" }, [
    el("div", { class: "row small" }, [el("b", {}, k), el("span", { class: "spacer" }), el("span", { class: "muted" }, (lv[k] || 0) + " 條")]),
    el("div", { class: "bar" }, el("span", { class: cls, style: "width:" + Math.round((lv[k] || 0) / tot * 100) + "%" }))]);
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "規則分級（自主審核閘門裁定）"),
    barRow("gold", "lvl-gold"), barRow("silver", "lvl-silver"), barRow("bronze", "lvl-bronze")]));

  const llm = await api.get("/api/llm/status");
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "LLM / 多智能體 後端"),
    el("div", { class: "kv" }, [el("b", {}, "後端"), el("span", {}, llm.backend + (llm.available ? "（真實模型）" : "（local 確定性）"))]),
    el("div", { class: "kv" }, [el("b", {}, "模型"), el("span", {}, llm.model)]),
    el("div", { class: "kv" }, [el("b", {}, "說明"), el("span", { class: "muted" }, llm.reason)]),
    el("p", { class: "notice" }, "無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。")]));
};

views.agent = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("智能體", "工具取證 · 回源核驗 · 安全治理。單智能體或多智能體合議。"));
  let mode = "council", role = "doctor";
  const chat = el("div", { class: "chat" });
  const input = el("textarea", { placeholder: "輸入問題，如：病人往來寒熱、胸脅苦滿、口苦，考慮什麼方？如何與大柴胡湯鑒別？" });
  const roleSel = el("select", {}, ["doctor", "researcher", "student", "patient"].map(r => el("option", { value: r }, { doctor: "醫師", researcher: "科研", student: "學生", patient: "患者" }[r])));
  roleSel.addEventListener("change", () => role = roleSel.value);
  const seg = el("div", { class: "seg" }, [
    el("button", { class: "on", onclick: e => { mode = "council"; setSeg(e); } }, "多智能體合議"),
    el("button", { onclick: e => { mode = "agent"; setSeg(e); } }, "單智能體"),
  ]);
  function setSeg(e) { [...seg.children].forEach(b => b.classList.remove("on")); e.target.classList.add("on"); }
  const sendBtn = el("button", { class: "btn", onclick: send }, "發送");

  async function send() {
    const q = input.value.trim(); if (!q) return;
    input.value = ""; sendBtn.disabled = true;
    chat.append(el("div", { class: "msg user" }, el("div", { class: "bubble" }, q)));
    const thinking = el("div", { class: "msg bot" }, el("div", { class: "bubble" }, el("span", { class: "muted" }, (mode === "council" ? "多智能體合議中…" : "智能體取證中…"))));
    chat.append(thinking); chat.scrollTop = chat.scrollHeight;
    let out;
    try { out = await api.post("/api/" + mode, { question: q, role }); }
    catch (e) { out = { answer: "請求失敗：" + e }; }
    thinking.remove();
    chat.append(renderBotAnswer(out, mode));
    chat.scrollTop = chat.scrollHeight; sendBtn.disabled = false;
  }
  input.addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send(); });

  main.append(el("div", { class: "agent-wrap" }, [
    el("div", { class: "agent-controls" }, [seg, el("span", { class: "muted small" }, "角色"), roleSel, el("span", { class: "spacer" }), el("span", { class: "badge backend " + (META.available ? "live" : "local") }, "後端：" + META.backend)]),
    chat,
    el("div", { class: "row", style: "margin-top:10px" }, [el("div", { style: "flex:1" }, input), sendBtn]),
    el("p", { class: "notice" }, "Ctrl/⌘ + Enter 發送。患者模式下涉及診斷/處方/劑量的問題將被安全攔截。"),
  ]));
  chat.append(el("div", { class: "msg bot" }, el("div", { class: "bubble" }, [
    el("div", { class: "answer-text" }, "您好，我是《傷寒論》智能體。我會調用工具檢索條文與規則，回答中的每條結論都會回源到條文編號。\n試試多智能體合議模式，可以看到「規劃→取證→方證/鑒別/六經/誤治專家→批評→綜合」的協作過程。") ])));
};

function renderBotAnswer(out, mode) {
  const wrap = el("div", { class: "msg bot" });
  const bub = el("div", { class: "bubble" });
  if (out.refused) {
    bub.append(el("div", { class: "refuse" }, [el("h4", {}, "已安全攔截（" + (out.refused_intents || []).join("、") + "）"), el("div", { class: "answer-text" }, out.message || "")]));
    wrap.append(bub); return wrap;
  }
  // trace / council timeline
  const steps = mode === "council" ? (out.council || []) : (out.agent_trace || []);
  if (steps.length) {
    const tr = el("div", { class: "trace" });
    steps.forEach(s => {
      if (mode === "council") {
        tr.append(el("div", { class: "step" }, [el("span", { class: "who" }, s.role_cn + "："), s.content + " ",
          ...(s.evidence_ids || []).slice(0, 4).map(clauseChip)]));
      } else {
        const k = s.kind, who = k === "tool_call" ? "工具 " + s.tool : k === "safety_block" ? "安全攔截" : k === "citation_check" ? "引用核驗" : k;
        tr.append(el("div", { class: "step " + (k === "tool_call" ? "tool" : "") }, [el("span", { class: "who" }, who + "："),
          k === "tool_call" ? JSON.stringify(s.arguments) : k === "citation_check" ? ((s.verified || []).length + " 條已核實") : (s.backend || "")]));
      }
    });
    bub.append(tr);
  }
  bub.append(el("div", { class: "answer-text" }, out.answer || "(無回答)"));
  const cr = out.citation_report;
  if (cr) {
    const ok = cr.ok && cr.has_any_citation;
    bub.append(el("div", { class: "cite-banner " + (ok ? "cite-ok" : "cite-warn") },
      ok ? "✓ 證據核驗通過 · 已核實條文 " + (cr.verified || []).join("、")
         : (cr.unsupported && cr.unsupported.length ? "⚠ 未能核實：" + cr.unsupported.join("、") : "⚠ 本回答未含可核驗條文編號")));
  }
  if (out.evidence_clause_ids && out.evidence_clause_ids.length)
    bub.append(el("div", { style: "margin-top:8px" }, clauseChips(out.evidence_clause_ids.slice(0, 12))));
  if (out.safety_notice) bub.append(el("div", { class: "notice" }, out.safety_notice));
  wrap.append(bub); return wrap;
}

views.search = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("原文檢索", "BM25 + 結構化過濾 + 關係圖譜擴展。所有命中回源 clause_id。"));
  const q = el("input", { type: "text", placeholder: "症狀 / 方名 / 脈象 / 治法，或「第38條」" });
  const ch = el("select", {}, [el("option", { value: "" }, "全部六經"), ...["太陽病", "陽明病", "少陽病", "太陰病", "少陰病", "厥陰病"].map(c => el("option", { value: c }, c))]);
  const expand = el("input", { type: "checkbox" });
  const out = el("div", {});
  async function run() {
    out.innerHTML = '<div class="loading">檢索中…</div>';
    const r = await api.post("/api/search", { query: q.value, six_channel: ch.value, expand: expand.checked, top_k: 10 });
    out.innerHTML = "";
    out.append(el("p", { class: "muted small" }, "命中 " + r.count + " 條"));
    (r.hits || []).forEach(h => out.append(el("div", { class: "card", style: "padding:12px 14px" }, [
      el("div", { class: "row small muted" }, [clauseChip(h.clause_id), el("span", {}, h.chapter + " · " + (h.six_channel || "")),
        el("span", { class: "layer " + layerLetter(h.layer_label) }, layerLetter(h.layer_label)), el("span", { class: "spacer" }), el("span", {}, "score " + h.score), h.match_source ? el("span", { class: "pill" }, h.match_source) : null]),
      el("p", { class: "classical", style: "margin:6px 0 0;font-size:14px" }, h.text),
      h.formulas && h.formulas.length ? el("div", {}, h.formulas.map(f => el("span", { class: "pill" }, f))) : null,
    ])));
  }
  q.addEventListener("keydown", e => { if (e.key === "Enter") run(); });
  main.append(el("div", { class: "card" }, [el("div", { class: "row" }, [el("div", { style: "flex:1" }, q), ch, el("label", { class: "row small" }, [expand, "關係擴展"]), el("button", { class: "btn", onclick: run }, "檢索")])]));
  main.append(out);
};

views.match = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("方證匹配", "醫師端輔助：依症狀/脈象匹配方證並回源條文（不替代臨床判斷）。"));
  const sym = chipInput("症狀，如 惡寒、發熱、無汗、身疼痛");
  const pul = chipInput("脈象，如 浮緊");
  const out = el("div", {});
  async function run() {
    out.innerHTML = '<div class="loading">匹配中…</div>';
    const r = await api.post("/api/match", { symptoms: sym.values(), pulse: pul.values(), top_k: 5 });
    out.innerHTML = "";
    if (r.safety_notice) out.append(el("div", { class: "cite-banner cite-ok" }, "⚕ " + r.safety_notice));
    (r.matched_formula_patterns || []).forEach(m => out.append(el("div", { class: "card" }, [
      el("div", { class: "row" }, [el("h3", {}, m.formula), el("span", { class: "pill" }, m.six_channel), el("span", { class: "spacer" }), el("span", { class: "badge" }, "匹配 " + m.match_score), el("span", { class: "pill" }, m.release_level)]),
      el("p", { class: "small" }, m.core_reason),
      m.matched_findings && m.matched_findings.length ? el("div", {}, m.matched_findings.map(f => el("span", { class: "pill" }, f))) : null,
      m.conflicts && m.conflicts.length ? el("div", { class: "cite-banner cite-warn" }, "衝突：" + m.conflicts.join("；")) : null,
      el("div", { style: "margin-top:8px" }, (m.evidence || []).map(e => el("div", { class: "evi" }, [el("div", { class: "ct" }, e.text), el("div", { class: "meta" }, [clauseChip(e.clause_id), " " + e.chapter]) ]))),
    ])));
    if (!(r.matched_formula_patterns || []).length) out.append(el("p", { class: "muted" }, "未找到顯著匹配。"));
  }
  main.append(el("div", { class: "card" }, [el("label", { class: "fld" }, [el("span", {}, "症狀"), sym.node]), el("label", { class: "fld" }, [el("span", {}, "脈象"), pul.node]), el("button", { class: "btn", onclick: run }, "匹配方證")]));
  main.append(out);
};

views.differential = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("方證鑒別", "選 2–3 方，生成多軸對比表與關鍵鑒別點。"));
  const fs = await api.get("/api/formulas");
  const sel = chipPicker(fs.formulas || [], "輸入或選擇方劑");
  const out = el("div", {});
  async function run() {
    const names = sel.values(); if (names.length < 2) { out.innerHTML = '<p class="muted">請至少選擇兩個方劑。</p>'; return; }
    out.innerHTML = '<div class="loading">分析中…</div>';
    const r = await api.post("/api/differential", { formulas: names });
    out.innerHTML = "";
    if (r.error) { out.append(el("p", { class: "muted" }, r.error)); return; }
    const d = r.differential;
    const tbl = el("table"); const head = el("tr", {}, [el("th", {}, "鑒別軸"), ...d.formulas.map(f => el("th", {}, f))]); tbl.append(head);
    (d.contrast_table || []).forEach(row => tbl.append(el("tr", {}, [el("td", {}, el("b", {}, row.axis)), ...d.formulas.map(f => el("td", {}, row[f] || "—"))])));
    out.append(el("div", { class: "card" }, [el("h3", {}, d.formulas.join(" vs ")), tbl]));
    if (d.key_discriminators) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "關鍵鑒別點"), ...d.key_discriminators.map(x => el("div", { class: "kv" }, [el("b", {}, "▸"), el("span", {}, x)]))]));
    // 十六輪：逐格回源核驗（規則歸類可錯——不再默認規則即正確）
    const v = r.verification;
    if (v) {
      const ok = !(v.flagged || []).length;
      out.append(el("div", { class: "cite-banner " + (ok ? "cite-ok" : "cite-warn") },
        ok ? "✓ 回源核驗通過：" + v.n_checked + " 項鑒別表述均可回源到支持條文"
           : "⚠ " + v.n_checked + " 項核驗中 " + v.flagged.length + " 項存疑（見下）"));
      if (!ok) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "存疑表述（規則層歸納，未能回源）"),
        ...v.flagged.map(f => el("div", { class: "kv small" }, [
          el("b", {}, f.formula + " · " + f.axis),
          el("span", {}, ["「" + f.term + "」",
            el("span", { class: "pill warn" }, f.status === "negated_context" ? "僅見否定語境（疑似歸類反了）" : "支持條文無此表述"),
            ...(f.clauses || []).map(clauseChip)])]))],
      ));
      if (v.note) out.append(el("p", { class: "small muted" }, v.note));
    }
    // 模型審校層（真模型：語義級對抗審校；local：確定性審校）
    const mr = r.model_review;
    if (mr) {
      const card = el("div", { class: "card" }, [
        el("div", { class: "section-title" }, ["模型審校 ", el("span", { class: "pill" }, mr.backend), el("span", { class: "pill " + (mr.verdict === "pass" ? "" : "warn") }, mr.verdict)]),
        el("p", { class: "small" }, mr.summary || "")]);
      (mr.issues || []).forEach(it => card.append(el("div", { class: "evi" }, [
        el("div", { class: "ct", style: "font-size:13px" }, (it.formula ? it.formula + " · " : "") + (it.axis ? it.axis + "：" : "") + it.problem),
        el("div", { class: "meta" }, [...(it.clause_ids || []).map(clauseChip),
          (it.unverified_clause_ids || []).length ? el("span", { class: "pill warn" }, "未核實引用：" + it.unverified_clause_ids.join("、")) : null])])));
      if (mr.note) card.append(el("p", { class: "small muted" }, mr.note));
      out.append(card);
    }
    if (d.supporting_clauses) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "證據條文"), clauseChips(d.supporting_clauses)]));
  }
  main.append(el("div", { class: "card" }, [sel.node, el("button", { class: "btn", style: "margin-top:10px", onclick: run }, "生成鑒別表")]));
  main.append(out);
};

views.teach = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("六經教學", "提綱 → 亞型 → 主方 → 誤治 → 禁忌 → 條文 → 練習題。"));
  const chans = await api.get("/api/channels");
  const sel = el("select", {}, (chans.channels || []).map(c => el("option", { value: c }, c)));
  const out = el("div", {});
  async function run() {
    out.innerHTML = '<div class="loading">編排中…</div>';
    const r = await api.post("/api/teach", { channel: sel.value });
    out.innerHTML = "";
    if (r.error) { out.append(el("p", { class: "muted" }, r.error)); return; }
    const L = r.lesson;
    const g = L["一、綱領"];
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "綱領"), el("p", { class: "classical" }, g.outline_text), el("div", { class: "row small" }, [clauseChip(g.outline_clause_id), g.resolution_time ? el("span", { class: "pill" }, "欲解時 " + g.resolution_time) : null]), el("p", { class: "small muted" }, g.summary)]));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "內部結構（亞型）"), ...L["二、內部結構（亞型）"].map(s => el("div", { class: "kv" }, [el("b", {}, s.name), el("span", {}, [el("span", { class: "muted small" }, "錨方："), (s.anchor_formulas || []).join("、"), " ", ...clauseChipsArr(s.evidence_clauses)])]))]));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "主要方劑"), ...L["三、主要方劑"].map(f => el("div", { class: "kv" }, [el("b", {}, f.formula), el("span", { class: "small" }, (f.core_symptoms || []).join("、"))]))]));
    const mist = L["四、誤治變證"] || []; if (mist.length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "誤治變證"), ...mist.map(m => el("div", { class: "kv small" }, [el("b", {}, "▸"), el("span", {}, m.path + "　" + (m.manifestations || []).join("、"))]))]));
    out.append(quizPanel(sel.value));
    if (r.safety_notice) out.append(el("p", { class: "notice" }, r.safety_notice));
  }
  main.append(el("div", { class: "card" }, [el("div", { class: "row" }, [sel, el("button", { class: "btn", onclick: run }, "生成課程")])]));
  main.append(out); run();
};

// 互動練習題面板（十八輪）：多題型題庫（seed 換批）+ 模型自主出題
function quizPanel(channel) {
  let seed = 1;
  const qBox = el("div", {});
  const status = el("span", { class: "small muted" });
  function renderQuestion(q) {
    const fb = el("div", { class: "small", style: "display:none;padding:6px 0" });
    const reveal = ok => {
      fb.style.display = "";
      fb.innerHTML = "";
      fb.append(el("span", { class: "pill " + (ok === false ? "warn" : "") },
        ok == null ? "答案" : ok ? "✓ 正確" : "✗ 錯誤"), " 答案：" + q.answer + "　");
      if (q.explanation) fb.append(el("span", { class: "muted" }, q.explanation + "　"));
      if (q.evidence_clause) fb.append(clauseChip(q.evidence_clause));
    };
    const opts = (q.options || []).length
      ? el("div", { class: "row", style: "flex-wrap:wrap;gap:4px;margin:4px 0" },
          q.options.map(o => el("button", { class: "btn ghost sm", onclick: e => {
            reveal(o === q.answer);
            e.target.classList.add(o === q.answer ? "on" : "off");
          } }, o)))
      : el("button", { class: "btn ghost sm", onclick: () => reveal(null) }, "顯示答案");
    return el("div", { style: "margin:10px 0;border-bottom:1px dashed var(--border);padding-bottom:8px" }, [
      el("div", {}, [el("b", {}, q.no + ". "), el("span", { class: "pill" }, q.type), " " + q.question]),
      opts, fb]);
  }
  async function load(useLlm) {
    qBox.innerHTML = '<div class="loading">' + (useLlm ? "模型命題中…" : "出卷中…") + "</div>";
    const r = await api.post("/api/quiz", { channel, n: 8, seed, use_llm: !!useLlm });
    qBox.innerHTML = "";
    status.textContent = "（" + (r.backend === "bank" ? "題庫 第" + seed + "批" : "後端 " + r.backend) + " · " + r.n + " 題）";
    (r.questions || []).forEach(q => qBox.append(renderQuestion(q)));
    if ((r.rejected_questions || []).length) qBox.append(el("p", { class: "small muted" },
      "⚠ 模型出的 " + r.rejected_questions.length + " 題因證據不合規被剔除（" + r.rejected_questions.map(x => x.reject_reason).join("；") + "）"));
    if (r.note) qBox.append(el("p", { class: "small muted" }, r.note));
  }
  const panel = el("div", { class: "card" }, [
    el("div", { class: "section-title" }, ["練習題（多題型 · 可換批 · 支持模型自主出題）", status]),
    el("div", { class: "row" }, [
      el("button", { class: "btn sm", onclick: () => { seed += 1; load(false); } }, "再出一批"),
      el("button", { class: "btn ghost sm", onclick: () => load(true) }, "模型自主出題"),
      el("span", { class: "small muted" }, "答案均錨定條文，點編號可回源")]),
    qBox]);
  load(false);
  return panel;
}

views.mistreatment = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("誤治傳變", "誤治方式 → 變證 → 救治方 → 原文證據。"));
  const r = await api.post("/api/mistreatment", {});
  const tbl = el("table"); tbl.append(el("tr", {}, [el("th", {}, "誤治"), el("th", {}, "變證"), el("th", {}, "救治方"), el("th", {}, "證據條文"), el("th", {}, "等級")]));
  (r.paths || []).forEach(p => tbl.append(el("tr", {}, [el("td", {}, p.mistreatment), el("td", {}, p.resulting_pattern), el("td", {}, (p.rescue_formulas || []).join("、")), el("td", {}, clauseChipsArr(p.clauses)), el("td", {}, el("span", { class: "pill" }, p.release_level))])));
  main.append(el("div", { class: "card" }, tbl));
};

views.research = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("科研挖掘", "共現網絡 · 頻次統計 · 家族樹 · 論文大綱。"));
  const topic = el("input", { type: "text", placeholder: "研究主題，如 桂枝湯類方證演化" });
  const out = el("div", {});
  async function run() {
    out.innerHTML = '<div class="loading">挖掘中…</div>';
    const r = await api.post("/api/research", { topic: topic.value || "全書方證" });
    out.innerHTML = "";
    // 頻次統計（真實數據，權重條視圖）
    const fq = r.frequency || {};
    const freqCol = (title, rows, unit) => {
      const max = rows.length ? rows[0][1] : 1;
      return el("div", { class: "card" }, [el("div", { class: "section-title" }, title),
        ...rows.slice(0, 15).map(([t, n]) => el("div", { style: "margin:3px 0" }, [
          el("div", { class: "row small" }, [el("b", {}, t), el("span", { class: "spacer" }), el("span", { class: "muted" }, n + unit)]),
          el("div", { class: "bar" }, el("span", { class: "lvl-gold", style: "width:" + Math.max(3, Math.round(n / max * 100)) + "%" }))]))]);
    };
    out.append(el("div", { class: "grid cols-3" }, [
      freqCol("高頻症狀", fq.symptom_frequency || [], " 條"),
      freqCol("高頻脈象", fq.pulse_frequency || [], " 條"),
      freqCol("高頻方劑", fq.formula_frequency || [], " 條"),
    ]));
    // 共現網絡（真實邊列表；主題聚焦時只看聚焦方）
    const nw = r.networks || {};
    const edges = (nw.focus_edges && nw.focus_edges.length ? nw.focus_edges : (nw.top_symptom_edges || [])).slice(0, 30);
    if (edges.length) {
      const maxW = edges[0].weight || 1;
      out.append(el("div", { class: "card" }, [
        el("div", { class: "section-title" }, "方-症共現網絡" + (nw.focus_formulas ? "（聚焦：" + nw.focus_formulas.join("、") + "）" : "（全書 Top 邊）")),
        el("p", { class: "small muted" }, "共 " + nw.formula_symptom_edges + " 條方-症邊 · " + nw.formula_pulse_edges + " 條方-脈邊；完整網絡已導出 " + (nw.files || []).join("、")),
        ...edges.map(e2 => el("div", { style: "margin:3px 0" }, [
          el("div", { class: "row small" }, [el("b", {}, e2.formula), el("span", { class: "muted" }, "——" + e2.symptom), el("span", { class: "spacer" }), el("span", { class: "muted" }, "共現 " + e2.weight + " 條")]),
          el("div", { class: "bar" }, el("span", { class: "lvl-silver", style: "width:" + Math.max(3, Math.round(e2.weight / maxW * 100)) + "%" }))]))]));
      if ((nw.top_pulse_edges || []).length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "方-脈共現（Top）"),
        el("div", {}, nw.top_pulse_edges.slice(0, 16).map(e2 => el("span", { class: "pill" }, e2.formula + "×" + e2.pulse + " " + e2.weight)))]));
    }
    // 家族樹（加減方演化）
    const ft = r.family_tree || {};
    if ((ft.families || []).length) out.append(el("div", { class: "card" }, [
      el("div", { class: "section-title" }, "方劑家族樹（" + ft.n_families + " 族" + (nw.focus_formulas ? " · 聚焦視圖" : "") + "）"),
      ...ft.families.map(fam => el("div", { style: "margin:6px 0" }, [
        el("b", {}, fam.base),
        ...(fam.modifications || []).map(m => el("div", { class: "kv small", style: "padding-left:14px" }, [
          el("b", {}, "└ " + m.modified_formula),
          el("span", { class: "muted" }, (m.added_herbs ? "＋" + m.added_herbs : "") + (m.removed_herbs ? "　－" + m.removed_herbs : "") || m.relation)]))])),
      el("p", { class: "small muted" }, ft.note || "")]));
    if (r.paper_outline) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "論文大綱：" + r.paper_outline.title),
      ...(r.paper_outline.sections || []).map(s => el("div", { class: "small" }, s)),
      el("p", { class: "small muted" }, "圖：" + (r.paper_outline.figures || []).join("、")),
      el("p", { class: "small muted" }, "表：" + (r.paper_outline.tables || []).join("、"))]));
    out.append(el("p", { class: "notice" }, r.safety_notice || ""));
  }
  main.append(el("div", { class: "card" }, [el("div", { class: "row" }, [el("div", { style: "flex:1" }, topic), el("button", { class: "btn", onclick: run }, "開始挖掘")])]));
  main.append(out);
};

views.paper = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("論文生成", "6 類論文 · 每處結論掛接規則/條文 ID · 含圖表源文件與 Cover Letter。"));
  const types = { formula_pattern: "方證規律挖掘", six_channel_kg: "六經知識圖譜", mistreatment: "誤治傳變研究", network_pharmacology: "網絡藥理前置", commentary_compare: "歷代注釋比較", methodology: "方法學研究" };
  const sel = el("select", {}, Object.keys(types).map(k => el("option", { value: k }, types[k])));
  const topic = el("input", { type: "text", placeholder: "主題（可選），如 桂枝湯類方" });
  const out = el("div", {});
  async function run() {
    out.innerHTML = '<div class="loading">撰寫中…（生成圖表與手稿）</div>';
    const r = await api.post("/api/paper", { type: sel.value, topic: topic.value });
    out.innerHTML = "";
    out.append(el("p", { class: "small muted" }, "已生成：" + r.manuscript_path));
    if (r.meta && r.meta.figures) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "圖表資產"), el("div", {}, [...(r.meta.figures || []), ...(r.meta.tables || [])].map(x => el("span", { class: "pill" }, x)))]));
    out.append(el("pre", { class: "md" }, r.manuscript));
  }
  main.append(el("div", { class: "card" }, [el("div", { class: "row" }, [sel, el("div", { style: "flex:1" }, topic), el("button", { class: "btn", onclick: run }, "生成論文")])]));
  main.append(out);
};

views.skills = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("Skill 庫", "編譯後的可調用技能（SKILL.md + rules.jsonl + examples.jsonl）。"));
  const r = await api.get("/api/skills");
  const groups = {};
  (r.skills || []).forEach(s => { const k = s.name.split(".").slice(0, 2).join("."); (groups[k] = groups[k] || []).push(s); });
  Object.keys(groups).sort().forEach(g => main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, g + "（" + groups[g].length + "）"),
    ...groups[g].map(s => el("div", { class: "kv small" }, [el("b", { style: "min-width:230px" }, s.name), el("span", { class: "muted" }, s.description)]))])));
};

views.about = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("接入 / 關於", "把本系統接入 Claude Code / Codex / OpenCode 等智能體框架。"));
  const llm = await api.get("/api/llm/status");
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "LLM 後端"),
    el("div", { class: "kv" }, [el("b", {}, "當前"), el("span", {}, llm.backend + " · " + llm.model)]),
    el("div", { class: "kv" }, [el("b", {}, "啟用真實模型"), el("span", { class: "small muted" }, "pip install litellm 並設置 ANTHROPIC_API_KEY（或 OPENAI_API_KEY 等），可選 HERMES_LLM_MODEL")]),
    el("div", {}, Object.entries(llm.recommended_models || {}).map(([k, v]) => el("div", { class: "kv small" }, [el("b", { style: "min-width:240px" }, k), el("span", { class: "muted" }, v)])))]));
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "接入智能體框架"),
    el("div", { class: "kv small" }, [el("b", {}, "Claude Code"), el("span", { html: "<code>claude mcp add shanghan -- python3 -m hermes_shanghan serve-mcp</code>" })]),
    el("div", { class: "kv small" }, [el("b", {}, "Codex/OpenCode"), el("span", { html: "<code>python3 -m hermes_shanghan export-tools --out tools.json</code>" })]),
    el("div", { class: "kv small" }, [el("b", {}, "工具調用"), el("span", { html: "<code>python3 -m hermes_shanghan tool-call shanghan_search --args '{\"query\":\"結胸\"}'</code>" })])]));
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "核心原則"),
    el("p", { class: "classical", style: "font-size:14px" }, "無原文，不成規則。無條文編號，不成證據。無證據鏈，不成回答。合併規則不能覆蓋初始條文規則。患者端禁止自動診斷、處方與劑量。")]));
};

// ---------- shared widgets ----------
function viewHead(t, p) {
  const tt = LANG === "en" ? (I18N_EN[t] || t) : t;
  return el("div", { class: "view-head" }, [el("h2", {}, tt), el("p", {}, p)]);
}
function clauseChipsArr(ids) { return (ids || []).map(clauseChip); }
function chipInput(placeholder) {
  const vals = [];
  const box = el("div", { class: "chip-input" });
  const inp = el("input", { type: "text", placeholder });
  function redraw() { [...box.querySelectorAll(".chip")].forEach(n => n.remove()); vals.forEach((v, i) => box.insertBefore(el("span", { class: "chip" }, [v, el("span", { class: "x", onclick: () => { vals.splice(i, 1); redraw(); } }, "✕")]), inp)); }
  function add() { const t = inp.value.trim().replace(/[，,、]$/, ""); if (t) { t.split(/[，,、]/).forEach(x => { x = x.trim(); if (x && !vals.includes(x)) vals.push(x); }); inp.value = ""; redraw(); } }
  inp.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "," || e.key === "，") { e.preventDefault(); add(); } });
  inp.addEventListener("blur", add);
  box.append(inp);
  return { node: box, values: () => { add(); return vals.slice(); } };
}
function chipPicker(options, placeholder) {
  const ci = chipInput(placeholder);
  const list = el("div", { class: "row", style: "margin-top:8px;max-height:120px;overflow:auto" });
  options.slice(0, 60).forEach(o => list.append(el("button", { class: "btn ghost sm", onclick: () => { ci.node.querySelector("input").value = o; ci.node.querySelector("input").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" })); } }, o)));
  return { node: el("div", {}, [ci.node, list]), values: ci.values };
}

// ---------- 溯源工作台（誤引檢測 / 文本回源 / 術語 / 方劑 / 爭議 / 比較） ----------
const TRACE_MODES = {
  quote: ["誤引檢測", "粘貼一段「引文」，逐片段判定：原文直引 / 後世歸納語 / 庫內無出處", "营卫不和，桂枝汤主之"],
  text: ["文本回源", "任意文本回源到條文（簡繁/異體字皆可）", "观其脉证，知犯何逆，随证治之"],
  term: ["術語譜系", "某術語是否原文？在庫首現注家與學派分佈", "營衛不和"],
  formula: ["方劑源流", "首見條文 → 方名傳播（含異名歸並）→ 歷代引用", "桂枝湯"],
  dispute: ["注家爭議", "條文號或條文文本句子 → 各家觀點 · 貼近原文程度 · 分歧類型（不裁決）；文本自動回源到最相近條文", "觀其脈證，知犯何逆，隨證治之"],
  compare: ["學派比較", "兩注家/學派對照：範式 · 指紋 · 一致度 · 高分歧條文", "柯琴 vs 尤怡"],
};
views.trace = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("溯源工作台", "引文審核 · 知識譜系追蹤。每個結論附證據層級標註。"));
  let mode = "quote";
  const seg = el("div", { class: "seg wrap" }, Object.keys(TRACE_MODES).map((k, i) =>
    el("button", { class: i === 0 ? "on" : "", onclick: e => { mode = k; [...seg.children].forEach(b => b.classList.remove("on")); e.target.classList.add("on"); inp.placeholder = TRACE_MODES[k][2]; hint.textContent = TRACE_MODES[k][1]; } }, TRACE_MODES[k][0])));
  const inp = el("textarea", { rows: 2, placeholder: TRACE_MODES.quote[2] });
  const hint = el("p", { class: "small muted" }, TRACE_MODES.quote[1]);
  const out = el("div", {});
  async function run() {
    const ref = inp.value.trim(); if (!ref) return;
    out.innerHTML = '<div class="loading">溯源中…</div>';
    const r = await api.post("/api/trace", { type: mode, ref });
    out.innerHTML = "";
    if (r.error) { out.append(el("div", { class: "cite-banner cite-warn" }, r.error)); return; }
    renderTrace(out, mode, r);
  }
  main.append(el("div", { class: "card" }, [seg, hint, inp, el("div", { class: "row", style: "margin-top:8px" }, [el("span", { class: "spacer" }), el("button", { class: "btn", onclick: run }, "追溯")])]));
  main.append(out);
};
function secBadge(t) { return el("span", { class: "pill" }, t); }
// 全庫候選出處可點閱：點擊即調 shanghan_library 讀取該書該章節原文
function libCandidateCard(h) {
  const body = el("div", { class: "lib-passage", style: "display:none" });
  let loaded = false;
  const head = el("div", { class: "kv small", style: "cursor:pointer", onclick: async () => {
    if (body.style.display === "none") {
      body.style.display = "";
      if (!loaded) {
        body.innerHTML = '<div class="loading">讀取《' + esc(h.title) + "》…</div>";
        try {
          const r = await api.post("/api/tool", { name: "shanghan_library", arguments: { book: h.book_id || h.title, section: h.section || "" } });
          body.innerHTML = "";
          if (r.error) body.append(el("p", { class: "muted small" }, r.error + (r.toc ? "　可用章節：" + r.toc.slice(0, 8).join("、") : "")));
          else {
            body.append(el("pre", { class: "md", style: "max-height:34vh;font-size:12.5px" }, r.text || ""));
            body.append(el("p", { class: "small muted" }, (r.truncated ? "（節選 " + (r.text || "").length + "/" + r.total_chars + " 字）" : "") + "　" + (r.evidence_layer || "")));
          }
          loaded = true;
        } catch (e) { body.innerHTML = '<p class="muted small">讀取失敗：' + esc(e.message) + "</p>"; }
      }
    } else body.style.display = "none";
  } }, [el("b", {}, "▸ " + h.title), el("span", {}, (h.author || "") + "·" + (h.dynasty || "") + (h.section ? " §" + h.section : "") + "　"),
    h.excerpt ? el("span", { class: "muted" }, "…" + h.excerpt.slice(0, 40) + "…") : null]);
  return el("div", {}, [head, body]);
}
// 歷代引用可點閱：點擊書名展開該書對相關條文的引用段落（分頁續讀）
function citingBookRow(b, citedIds, label) {
  const body = el("div", { class: "lib-passage", style: "display:none" });
  let offset = 0, open = false;
  async function loadMore() {
    const loading = el("div", { class: "loading" }, "讀取引用段落…");
    body.append(loading);
    try {
      const r = await api.post("/api/trace/passages", { book_dir: b.book_dir || b.book, clause_ids: citedIds || [], offset, limit: 6 });
      loading.remove();
      (r.passages || []).forEach(p => body.append(el("div", { class: "evi" }, [
        el("div", { class: "ct", style: "font-size:12px" }, p.excerpt || p.matched_span),
        el("div", { class: "meta" }, [el("span", { class: "pill" }, p.mode), clauseChip(p.clause_id), " §" + p.chapter + " · 覆蓋 " + p.coverage,
          p.via_commentator ? el("span", { class: "muted" }, "　經" + p.via_commentator + "注轉引") : null])])));
      offset += (r.passages || []).length;
      const old = body.querySelector(".btn.more"); if (old) old.remove();
      if (r.has_more) body.append(el("button", { class: "btn ghost sm more", onclick: loadMore }, "載入更多（共 " + r.n_passages + " 段）"));
    } catch (e) { loading.textContent = "讀取失敗：" + e.message; }
  }
  const head = el("div", { class: "kv small", style: "cursor:pointer", onclick: () => {
    open = !open;
    body.style.display = open ? "" : "none";
    if (open && !offset && !body.childElementCount) loadMore();
  } }, [el("b", {}, "▸ " + (b.book || b.book_dir)), el("span", {}, label || ((b.author ? b.author + "　" : "") + (b.n_paragraphs != null ? b.n_paragraphs + " 段引用" : "")))]);
  return el("div", {}, [head, body]);
}
function citationsByDynasty(cit, title) {
  const card = el("div", { class: "card" }, [el("div", { class: "section-title" }, title + "（" + (cit.n_citing_books || 0) + " 部書 · 點擊書名展開引用段落）")]);
  (cit.by_dynasty || []).forEach(d => {
    card.append(el("div", { class: "small", style: "margin-top:6px" }, el("b", {}, d.dynasty)));
    (d.books || []).forEach(b => card.append(citingBookRow(b, cit.cited_clause_ids)));
  });
  return card;
}
function renderModelSynthesis(out, r) {
  const ms = r.model_synthesis;
  if (!ms || ms.error) return;
  const cr = ms.citation_report;
  out.append(el("div", { class: "card" }, [
    el("div", { class: "section-title" }, ["模型綜合 ", el("span", { class: "pill" }, ms.backend), el("span", { class: "pill" }, ms.evidence_layer || "")]),
    el("div", { class: "answer-text", style: "font-size:13.5px" }, ms.synthesis || ""),
    cr ? el("div", { class: "cite-banner " + (cr.ok ? "cite-ok" : "cite-warn") },
      cr.ok ? "✓ 綜合中引用已全部核驗（" + (cr.verified || []).length + " 條）"
            : "⚠ 引用核驗未全部通過：" + [...(cr.unsupported || []), ...(cr.outside_evidence || [])].join("、")) : null,
    ms.note ? el("p", { class: "small muted" }, ms.note) : null]));
}
function renderTrace(out, mode, r) {
  renderModelSynthesis(out, r);
  if (mode === "quote") {
    const ok = (r.verdict || "").includes("可作原文直引");
    out.append(el("div", { class: "cite-banner " + (ok ? "cite-ok" : "cite-warn") }, r.verdict || ""));
    (r.fragments || []).forEach(f => {
      const cls = f.verdict === "原文逐字" ? "frag-a" : f.verdict.includes("後世") ? "frag-c" : "frag-x";
      out.append(el("div", { class: "card frag " + cls }, [
        el("div", { class: "row" }, [el("b", { class: "classical" }, f.fragment), el("span", { class: "pill" }, f.verdict)]),
        f.verbatim_in && f.verbatim_in.length ? clauseChips(f.verbatim_in) : null,
        (f.related_claims || []).map(c => el("div", { class: "small muted" }, "關聯觀點 " + c.claim_id + "（" + c.evidence_grade + "）：原文相關表達 " + Object.keys(c.related_original_terms || {}).join("、"))),
        f.posthoc_terms && f.posthoc_terms.length ? el("div", { class: "small muted" }, "後世術語：" + f.posthoc_terms.join("、")) : null,
      ]));
    });
    (r.warnings || []).forEach(w => out.append(el("p", { class: "notice" }, w)));
  } else if (mode === "term") {
    out.append(el("div", { class: "cite-banner cite-ok" }, r.citable_statement || r.evidence_grade || ""));
    if ((r.verbatim_in_original || []).length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "原文逐字出現"), clauseChips(r.verbatim_in_original)]));
    const tbl = el("table"); tbl.append(el("tr", {}, ["朝代", "注家", "注本", "學派", "用例"].map(h => el("th", {}, h))));
    (r.commentarial_chronology || []).forEach(e2 => tbl.append(el("tr", {}, [el("td", {}, e2.dynasty), el("td", {}, e2.commentator), el("td", {}, e2.book), el("td", {}, e2.school_id || ""), el("td", {}, e2.n_passages + " 段 " ), ])));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "注家使用譜（在庫九注本）"), tbl]));
  } else if (mode === "dispute") {
    if (r.resolved_from_text) out.append(el("div", { class: "cite-banner cite-ok" },
      "已由文本回源到 " + r.resolved_from_text.matched_clause_id + "（片段 " + r.resolved_from_text.longest_run + " 字）" +
      ((r.resolved_from_text.alternatives || []).length ? "　備選：" + r.resolved_from_text.alternatives.join("、") : "")));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, [clauseChip(r.clause.clause_id), " 條文"]), el("p", { class: "classical" }, r.clause.text)]));
    const tbl = el("table"); tbl.append(el("tr", {}, ["朝代", "注家", "出處", "學派", "貼近原文", "後世術語", "取徑", "節錄"].map(h => el("th", {}, h))));
    (r.views || []).forEach(v => tbl.append(el("tr", {}, [el("td", {}, v.dynasty), el("td", {}, v.commentator), el("td", { class: "small" }, "《" + v.book + "》" + (v.chapter ? "·" + v.chapter : "")), el("td", { class: "small" }, v.school), el("td", {}, String(v.closeness_to_original)), el("td", { class: "small" }, (v.posthoc_terms || []).join("、")), el("td", { class: "small" }, (v.analytic_focus || []).join("/")), el("td", { class: "small muted" }, v.excerpt)])));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "各家觀點（" + r.n_commentators + " 家 · 分歧類型：" + (r.divergence_types_present || []).join("、") + "）"), el("div", { class: "tbl-scroll" }, tbl)]));
    out.append(el("p", { class: "notice" }, r.undecidable_note));
  } else if (mode === "compare") {
    out.append(el("div", { class: "grid cols-2" }, [r.a, r.b].map(s => el("div", { class: "card" }, [
      el("h3", {}, s.name), el("div", { class: "kv small" }, [el("b", {}, "學派"), el("span", {}, s.school)]),
      el("div", { class: "kv small" }, [el("b", {}, "範式"), el("span", {}, s.paradigm)]),
      s.fingerprint_terms && s.fingerprint_terms.length ? el("div", {}, s.fingerprint_terms.map(secBadge)) : null]))));
    if (r.agreement) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "實測一致度"), el("p", {}, "共注條文 " + r.agreement.n_shared_clauses + " · 術語一致度 " + r.agreement.mean_term_agreement)]));
    if ((r.top_divergent_clauses || []).length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "高分歧條文"), ...r.top_divergent_clauses.map(t => el("div", { class: "kv small" }, [clauseChip(t.clause_id), el("span", {}, "分歧 " + t.term_divergence)]))]));
    out.append(el("p", { class: "notice" }, r.reading_note || ""));
  } else if (mode === "formula") {
    const fa = r.first_attestation || {};
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "首見（宋本條文序）"), el("div", { class: "row" }, [clauseChip(fa.clause_id), el("span", { class: "small muted" }, fa.note)]), el("p", { class: "small" }, fa.core_pattern)]));
    const nt = r.name_transmission || {};
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "方名傳播"), el("p", {}, nt.total_mentions + " 次 / " + nt.n_books + " 部書"),
      ...(nt.aliases || []).map(a => el("div", { class: "evi" }, [el("div", { class: "ct" }, "異名「" + a.alias + "」" + (a.same_formula ? "（同方）" : "（不可合併）") + "：" + a.alias_mentions + " 次 / " + a.alias_n_books + " 部書"), el("div", { class: "meta" }, a.source)]))]));
    const cit = r.citations_of_clauses || {};
    out.append(citationsByDynasty(cit, "歷代引用"));
    (r.claims || []).forEach(c => out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "方證觀點 " + c.claim_id), el("p", {}, c.claim), el("span", { class: "pill" }, c.evidence_grade)])));
  } else { // text / claim / fallback
    if (r.matches) {
      out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "回源命中"), ...r.matches.map(m => el("div", { class: "kv small" }, [clauseChip(m.clause_id), el("span", {}, "片段 " + (m.longest_run || 0) + " 字 · 覆蓋 " + (m.coverage || 0))]))]));
      if (!r.matches.length) {
        out.append(el("p", { class: "muted" }, r.note || ""));
        const lc = r.library_candidates;
        if (lc && (lc.hits || []).length) out.append(el("div", { class: "card" }, [
          el("div", { class: "section-title" }, "全庫候選出處（旁證 · 點擊展開原文）"),
          ...lc.hits.map(libCandidateCard),
          el("p", { class: "small muted" }, lc.note || "")]));
        else if (lc && lc.available === false) out.append(el("p", { class: "small muted" }, lc.note || "全庫未下載"));
      }
    }
    if (r.clause) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, [clauseChip(r.clause.clause_id), " 原文"]), el("p", { class: "classical" }, r.clause.text)]));
    if (r.claim) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, r.claim_id || "方證觀點"), el("p", {}, r.claim), el("span", { class: "pill" }, r.evidence_grade)]));
    if (r.citations) out.append(citationsByDynasty(r.citations, "歷代引用"));
    if (r.main_path) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "主路徑"), el("p", { class: "small" }, r.main_path.map(p => p.dynasty + "·" + p.book).join(" → "))]));
  }
  if (r.section_evidence_levels) out.append(el("details", { class: "card small" }, [el("summary", {}, "逐節證據層級"), ...Object.entries(r.section_evidence_levels).map(([k, v]) => el("div", { class: "kv small" }, [el("b", {}, k), el("span", { class: "muted" }, v)]))]));
}

// ---------- 方藥檔案（藥證 / 方解） ----------
views.herbs = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("方藥檔案", "藥證檔案（A-derived，不編造藥性）與方解一站式（四層症狀口徑）。"));
  const hIn = el("input", { type: "text", placeholder: "藥名，如 桂枝" });
  const fIn = el("input", { type: "text", placeholder: "方名，如 桂枝湯" });
  const out = el("div", {});
  async function runHerb() {
    out.innerHTML = '<div class="loading">生成藥證檔案…</div>';
    const herb = hIn.value.trim();
    const r = await api.post("/api/tool", { name: "shanghan_herb_profile", arguments: { herb } });
    out.innerHTML = "";
    if (r.error) { out.append(el("p", { class: "muted" }, r.error)); return; }
    // 條文分頁：載入更多續讀（十七輪）
    const chipBox = el("div", {}); (r.clause_ids || []).forEach(id => chipBox.append(clauseChip(id)));
    let cOff = (r.clause_ids || []).length;
    const moreClauses = el("button", { class: "btn ghost sm", style: r.clauses_has_more ? "" : "display:none", onclick: async () => {
      moreClauses.disabled = true;
      const r2 = await api.post("/api/tool", { name: "shanghan_herb_profile", arguments: { herb, clause_offset: cOff, clause_limit: 20 } });
      (r2.clause_ids || []).forEach(id => chipBox.append(clauseChip(id)));
      cOff += (r2.clause_ids || []).length;
      moreClauses.disabled = false;
      if (!r2.clauses_has_more) moreClauses.style.display = "none";
    } }, "載入更多條文（共 " + r.n_clauses + " 條）");
    out.append(el("div", { class: "grid cols-2" }, [
      el("div", { class: "card" }, [el("div", { class: "section-title" }, r.herb + " · 出現"), el("p", {}, r.n_formulas + " 方 · " + r.n_clauses + " 條"), chipBox, moreClauses]),
      el("div", { class: "card" }, [el("div", { class: "section-title" }, "配伍共現（同方計數）"), ...(r.top_partners || []).map(p => el("div", { class: "kv small" }, [el("b", {}, p.herb), el("span", {}, p.n_formulas_together + " 方同用")]))]),
    ]));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "原文劑量寫法"), el("div", {}, (r.dose_variants || []).map(secBadge))]));
    // 本草層：摘錄可點擊展開全節原文 + 分頁續讀更多本草書
    const bc = r.bencao_layer || {};
    const bcBox = el("div", {});
    const renderBc = e2 => bcBox.append(
      el("div", { class: "evi" }, [el("div", { class: "ct" }, e2.excerpt),
        el("div", { class: "meta" }, "《" + e2.book + "》" + e2.author + "·" + e2.dynasty + (e2.nature_flavor ? "　味" + e2.nature_flavor.flavor + "·" + e2.nature_flavor.nature : ""))]),
      libCandidateCard({ book_id: e2.book_id, title: e2.book, author: e2.author, dynasty: e2.dynasty, section: e2.section }));
    (bc.excerpts || []).forEach(renderBc);
    let bOff = (bc.excerpts || []).length;
    const moreBc = el("button", { class: "btn ghost sm", style: bc.has_more ? "" : "display:none", onclick: async () => {
      moreBc.disabled = true;
      const r2 = await api.post("/api/tool", { name: "shanghan_herb_profile", arguments: { herb, bencao_offset: bOff, bencao_limit: 4 } });
      const bc2 = r2.bencao_layer || {};
      (bc2.excerpts || []).forEach(renderBc);
      bOff += (bc2.excerpts || []).length;
      moreBc.disabled = false;
      if (!bc2.has_more) moreBc.style.display = "none";
    } }, "更多本草摘錄");
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "本草層（旁證 · ▸ 展開該書該節原文）"),
      bc.available ? bcBox : el("p", { class: "muted small" }, bc.note || ""), bc.available ? moreBc : null,
      bc.available && bc.note ? el("p", { class: "small muted" }, bc.note) : null]));
    (r.warnings || []).forEach(w => out.append(el("p", { class: "notice" }, w)));
  }
  async function runFormula() {
    out.innerHTML = '<div class="loading">生成方解…</div>';
    const r = await api.post("/api/formula-explain", { name: fIn.value.trim() });
    out.innerHTML = "";
    if (r.error) { out.append(el("p", { class: "muted" }, r.error)); return; }
    const L = r.symptom_layers || {};
    const layerCard = (title, body) => el("div", { class: "card" }, [el("div", { class: "section-title" }, title), body]);
    out.append(el("div", { class: "grid cols-2" }, [
      layerCard("① 首見方證（" + (L.first_attestation || {}).clause_id + "）", el("div", {}, [...((L.first_attestation || {}).symptoms || []).map(secBadge), ...(((L.first_attestation || {}).pulse || []).map(p => el("span", { class: "pill" }, "脈" + p)))])),
      layerCard("② 規則歸納核心證（D 層）", el("div", {}, ((L.rule_induced_core || {}).symptoms || []).map(secBadge))),
      layerCard("③ 全書聚合（含頻次）", el("div", {}, (L.aggregate_all_clauses || []).slice(0, 12).map(s => el("span", { class: "pill" }, s.symptom + "×" + s.n_clauses)))),
      layerCard("④ 特殊上下文（誤治/禁忌/傳變）", el("div", {}, (L.special_context || []).map(s => el("div", { class: "kv small" }, [clauseChip(s.clause_id), el("span", {}, "[" + s.context.join("/") + "] " + (s.symptoms || []).join("、"))])))),
    ]));
    out.append(el("p", { class: "notice" }, L.note || ""));
    const adm = r.administration || {};
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "煎服法（原文）"),
      adm.preparation ? el("div", { class: "kv small" }, [el("b", {}, "煎法"), el("span", {}, adm.preparation)]) : null,
      adm.administration ? el("div", { class: "kv small" }, [el("b", {}, "服法"), el("span", {}, adm.administration)]) : null,
      el("div", { class: "cite-banner cite-warn" }, adm.warning || "")]));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "組成"), el("div", { class: "small" }, (r.composition || []).map(x => x.herb + (x.dose_processing ? "（" + x.dose_processing + "）" : "")).join("　"))]));
    if ((r.differentials || []).length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "類方鑒別"), ...r.differentials.map(d => el("div", { class: "kv small" }, [el("b", {}, "vs " + d.vs.join("、")), el("span", {}, (d.key_discriminators || []).join("；"))]))]));
  }
  main.append(el("div", { class: "grid cols-2" }, [
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "藥證檔案"), el("div", { class: "row" }, [el("div", { style: "flex:1" }, hIn), el("button", { class: "btn", onclick: runHerb }, "生成")])]),
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "方解一站式"), el("div", { class: "row" }, [el("div", { style: "flex:1" }, fIn), el("button", { class: "btn", onclick: runFormula }, "生成")])]),
  ]));
  main.append(out);
};

// ---------- 辨證閉環（四診採集 / 裁決 / 衝突審計） ----------
views.bianzheng = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("辨證閉環", "四診採集（信息整理）→ 多假設裁決 → 方證衝突審計。醫師/教學輔助，不替代臨床。"));
  const txt = el("textarea", { rows: 2, placeholder: "自然敘述，如：发热，怕冷，出汗，头痛，服退烧药后腹泻" });
  const out = el("div", {});
  async function runIntake() {
    out.innerHTML = '<div class="loading">整理中…</div>';
    const r = await api.post("/api/intake", { text: txt.value });
    out.innerHTML = "";
    const rows = [["主訴", r.chief_complaint], ["病程", (r.timeline || []).join("、")], ["寒熱", (r.cold_heat || []).join("、")], ["汗", (r.sweating || []).join("、")], ["渴飲", (r.thirst_drinking || []).join("、")], ["二便", (r.stool_urine || []).join("、")], ["胸脅", (r.chest_hypochondrium || []).join("、")], ["心腹", (r.epigastrium_abdomen || []).join("、")], ["痛", (r.pain_location || []).join("、")], ["眠", (r.sleep || []).join("、")], ["脈", (r.pulse || []).join("、")], ["誤治史", (r.prior_mistreatment || []).join("、")], ["藥後", (r.medication_response || []).join("、")]];
    const tbl = el("table"); rows.forEach(([k, v]) => tbl.append(el("tr", {}, [el("th", {}, k), el("td", {}, v || "—")])));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "結構化四診表"), el("div", { class: "tbl-scroll" }, tbl)]));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "缺失關鍵信息 → 追問"), ...(r.next_questions || []).map(q => el("div", { class: "kv small" }, [el("b", {}, "?"), el("span", {}, q)]))]));
    // 模型輔助抽取（十七輪：規則詞表之上的語義層，逐詞回驗敘述原文）
    const mx = r.model_extraction;
    if (mx && !mx.error) {
      out.append(el("div", { class: "card" }, [
        el("div", { class: "section-title" }, ["模型輔助抽取 ", el("span", { class: "pill" }, mx.backend)]),
        (mx.added_findings || []).length ? el("div", { class: "kv small" }, [el("b", {}, "補充表現"), el("span", {}, mx.added_findings.map(secBadge))]) : el("p", { class: "small muted" }, "無規則詞表之外的補充"),
        (mx.model_pulse || []).length ? el("div", { class: "kv small" }, [el("b", {}, "脈象"), el("span", {}, mx.model_pulse.join("、"))]) : null,
        (mx.unverified || []).length ? el("div", { class: "cite-banner cite-warn" }, "⚠ 敘述中找不到依據、未併入：" + mx.unverified.join("、")) : null,
        el("p", { class: "small muted" }, mx.note || "")]));
    }
    out.append(el("p", { class: "notice" }, r.note || ""));
    // 醫師端一鍵送裁決（含模型補充的已驗證表現）
    const allSyms = ["cold_heat", "sweating", "thirst_drinking", "stool_urine", "chest_hypochondrium", "epigastrium_abdomen", "pain_location", "sleep", "other_findings"].flatMap(k => r[k] || [])
      .concat((mx && mx.added_findings) || []);
    if (allSyms.length) out.append(el("button", { class: "btn", onclick: () => runAdj(allSyms, (r.pulse && r.pulse.length ? r.pulse : (mx && mx.model_pulse) || [])) }, "→ 送入多假設裁決（醫師端）"));
  }
  async function runAdj(symptoms, pulse) {
    out.append(el("div", { class: "loading" }, "裁決中…"));
    const r = await api.post("/api/adjudicate", { symptoms, pulse });
    out.append(el("div", { class: "cite-banner " + (r.verdict && r.verdict.startsWith("傾向") ? "cite-ok" : "cite-warn") }, "裁決：" + r.verdict + " —— " + r.rationale));
    (r.candidates || []).forEach(h => out.append(el("div", { class: "card" }, [
      el("div", { class: "row" }, [el("h3", {}, h.formula), el("span", { class: "pill" }, "信心 " + (h.confidence || ""))]),
      el("div", { class: "kv small" }, [el("b", {}, "支持"), el("span", {}, (h.support || []).join("、") || "—")]),
      el("div", { class: "kv small" }, [el("b", {}, "反證"), el("span", {}, (h.against || []).join("、") || "—")]),
      el("div", { class: "kv small" }, [el("b", {}, "缺失"), el("span", {}, (h.missing_key_findings || []).join("、") || "—")]),
      (h.contraindication_hits || []).length ? el("div", { class: "cite-banner cite-warn" }, "禁忌衝突：" + h.contraindication_hits.map(c => c.presented).join("、")) : null,
    ])));
    (r.key_questions || []).forEach(q => out.append(el("p", { class: "notice" }, "追問：" + q)));
    // 模型審校（十七輪：規則裁決之上的語義層——漏診方向/裁決穩妥性/追問）
    const mr = r.model_review;
    if (mr) {
      const card = el("div", { class: "card" }, [
        el("div", { class: "section-title" }, ["裁決模型審校 ", el("span", { class: "pill" }, mr.backend),
          mr.agrees_with_verdict === false ? el("span", { class: "pill warn" }, "對裁決有異議") : null]),
        el("p", { class: "small" }, mr.assessment || "")]);
      (mr.missed_patterns || []).forEach(mp => card.append(el("div", { class: "evi" }, [
        el("div", { class: "ct", style: "font-size:13px" }, "漏診方向：" + mp.formula + " —— " + mp.reason),
        el("div", { class: "meta" }, [...(mp.clause_ids || []).map(clauseChip),
          (mp.unverified_clause_ids || []).length ? el("span", { class: "pill warn" }, "未核實引用：" + mp.unverified_clause_ids.join("、")) : null])])));
      (mr.additional_questions || []).forEach(q => card.append(el("p", { class: "notice" }, "模型追問：" + q)));
      const cr = mr.citation_report;
      if (cr && !cr.ok) card.append(el("div", { class: "cite-banner cite-warn" }, "⚠ 審校意見中存在未核實引用：" + [...(cr.unsupported || []), ...(cr.outside_evidence || [])].join("、")));
      if (mr.note) card.append(el("p", { class: "small muted" }, mr.note));
      out.append(card);
    }
  }
  const cf = el("input", { type: "text", placeholder: "方名，如 桂枝湯" });
  const cs = el("input", { type: "text", placeholder: "呈現表現（逗號分隔），如 無汗,發熱" });
  async function runConflict() {
    out.innerHTML = '<div class="loading">審計中…</div>';
    const r = await api.post("/api/tool", { name: "shanghan_conflict_audit", arguments: { formula: cf.value.trim(), symptoms: cs.value.split(/[,，、]/).map(s => s.trim()).filter(Boolean) } });
    out.innerHTML = "";
    if (r.error) { out.append(el("p", { class: "muted" }, r.error)); return; }
    out.append(el("div", { class: "cite-banner " + (r.severity === "無衝突" ? "cite-ok" : "cite-warn") }, r.formula + " · 衝突嚴重度：" + r.severity));
    (r.conflicts || []).forEach(c => out.append(el("div", { class: "card" }, [el("div", { class: "row" }, [el("b", {}, "呈現「" + c.presented + "」 vs 方證期望「" + c.pattern_expects + "」"), el("span", { class: "pill" }, c.strength)]), clauseChips(c.supporting_clauses)])));
    if ((r.reassign_candidates || []).length) out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "改判候選（定位提示）"), ...r.reassign_candidates.map(a => el("div", { class: "kv small" }, [el("b", {}, a.candidate), el("span", {}, "因「" + a.conflict + "」")]))]));
    (r.should_ask || []).forEach(q => out.append(el("p", { class: "notice" }, "應補問：" + q)));
  }
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "① 四診信息採集（患者端安全：只整理不診斷）"), txt, el("div", { class: "row", style: "margin-top:8px" }, [el("span", { class: "spacer" }), el("button", { class: "btn", onclick: runIntake }, "整理四診表")])]));
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "② 方證衝突審計（醫師端）"), el("div", { class: "row" }, [cf, el("div", { style: "flex:1" }, cs), el("button", { class: "btn", onclick: runConflict }, "審計")])]));
  main.append(out);
};

// ---------- 工具台（28 工具通用調用 / 全庫 / 深度研究 / 評測 / 標註閉環） ----------
views.tools = async (main) => {
  main.innerHTML = "";
  main.append(viewHead("工具台", "28 個回源工具通用調用 · 笈成全庫 · 深度研究循環 · 評測看板 · 金標準標註閉環。"));
  const out = el("div", {});

  // -- 通用工具調用 --
  const specs = (await api.get("/api/tools")).tools || [];
  const sel = el("select", {}, specs.map(s => el("option", { value: s.function.name }, s.function.name)));
  const desc = el("p", { class: "small muted" });
  const argsBox = el("textarea", { rows: 3 });
  function fillTemplate() {
    const s = specs.find(x => x.function.name === sel.value); if (!s) return;
    desc.textContent = s.function.description;
    const props = (s.function.parameters || {}).properties || {};
    const tpl = {};
    Object.keys(props).forEach(k => { tpl[k] = props[k].default !== undefined ? props[k].default : (props[k].type === "array" ? [] : props[k].type === "integer" ? 0 : ""); });
    argsBox.value = JSON.stringify(tpl, null, 1);
  }
  sel.addEventListener("change", fillTemplate); fillTemplate();
  async function runTool() {
    out.innerHTML = '<div class="loading">調用中…</div>';
    let args; try { args = JSON.parse(argsBox.value || "{}"); } catch (e) { out.innerHTML = '<p class="muted">參數不是合法 JSON</p>'; return; }
    const r = await api.post("/api/tool", { name: sel.value, arguments: args });
    out.innerHTML = "";
    if (r.evidence_level) out.append(el("div", { class: "row small" }, [el("span", { class: "pill" }, "證據層 " + r.evidence_level), r.confidence != null ? el("span", { class: "pill" }, "置信 " + r.confidence) : null]));
    out.append(el("pre", { class: "md tbl-scroll" }, JSON.stringify(r, null, 1).slice(0, 12000)));
  }
  main.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "通用工具調用（結果附證據層標註）"),
    el("div", { class: "row" }, [sel, el("button", { class: "btn", onclick: runTool }, "調用")]), desc, argsBox]));

  // -- 笈成全庫 --
  const libQ = el("input", { type: "text", placeholder: "書名/作者（編目）或原文詞句（全文），如 奔豚" });
  async function runLib() {
    out.innerHTML = '<div class="loading">全庫檢索中…</div>';
    const r = await api.post("/api/tool", { name: "shanghan_library", arguments: { query: libQ.value.trim(), top_k: 6 } });
    out.innerHTML = "";
    if (r.available === false) { out.append(el("div", { class: "cite-banner cite-warn" }, r.hint || "全庫未下載")); return; }
    (r.catalog_hits || []).length && out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "編目命中"), ...r.catalog_hits.map(h => el("div", { class: "kv small" }, [el("b", {}, h.id), el("span", {}, (h.author || "") + "·" + (h.dynasty || "") + " [" + (h.category || "") + "]")]))]));
    (r.text_hits || []).length && out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "全文命中（" + r.n_text_hits + "）"), ...r.text_hits.map(h => el("div", { class: "evi" }, [el("div", { class: "ct" }, "…" + h.excerpt + "…"), el("div", { class: "meta" }, "《" + h.title + "》" + (h.author || "") + "·" + (h.dynasty || "") + " §" + h.section)]))]));
    out.append(el("p", { class: "notice" }, r.evidence_layer || ""));
  }

  // -- 深度研究 --
  const drQ = el("input", { type: "text", placeholder: "研究主題，如 桂枝湯類方的劑量演化" });
  async function runDR() {
    out.innerHTML = '<div class="loading">深度研究循環運行中（規劃→子代理→批評家）…</div>';
    const r = await api.post("/api/deep-research", { topic: drQ.value.trim() || "桂枝湯類方源流", rounds: 3 });
    out.innerHTML = "";
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "七維覆蓋（" + r.n_rounds + " 輪 · 後端 " + r.backend + "）"),
      el("div", {}, Object.entries(r.coverage || {}).map(([d, n]) => el("span", { class: "pill" }, d + " ×" + n)))]));
    (r.findings || []).forEach(f => out.append(el("div", { class: "card" }, [
      el("div", { class: "row small" }, [el("span", { class: "pill" }, f.dimension), el("b", {}, f.module), el("span", { class: "spacer" }), el("span", { class: f.citation_ok ? "pill" : "pill warn" }, f.citation_ok ? "引用核驗✓" : "核驗⚠")]),
      el("p", { class: "small" }, f.summary), clauseChips((f.verified_clause_ids || []).slice(0, 6))])));
  }

  // -- 評測看板 --
  async function runEval() {
    out.innerHTML = '<div class="loading">讀取評測指標…</div>';
    const r = await api.post("/api/tool", { name: "shanghan_eval_metrics", arguments: {} });
    out.innerHTML = "";
    const suites = r.suites || {};
    const rows = [];
    const cz = (suites.cloze || {}).metrics || {}; if (cz.attainable) rows.push(["遮方預測（可達折）", "Top-1 " + cz.attainable.top1 + " · Top-3 " + cz.attainable.top3 + " · MRR " + cz.attainable.mrr]);
    const cs = (suites.cases || {}).metrics || {}; if (cs.top1 != null) rows.push(["醫案回放", "Top-1 " + cs.top1 + " · MRR " + (cs.mrr || "")]);
    const gr = (suites.grounding || {}).metrics || {}; if (gr.grounded_answer_rate != null) rows.push(["證據接地率", gr.grounded_answer_rate]);
    const ab = (suites.agent || {}).metrics || {}; if (ab.routing_accuracy != null) rows.push(["智能體路由", ab.routing_accuracy]);
    const tbl = el("table"); rows.forEach(([k, v]) => tbl.append(el("tr", {}, [el("th", {}, k), el("td", {}, String(v))])));
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "四大基準（確定性 · 零人工標註）"), el("div", { class: "tbl-scroll" }, tbl), el("p", { class: "small muted" }, "缺項請先運行 evaluate；引文識別自檢見 docs/TRACE.md")]));
  }

  // -- 金標準標註閉環（human-in-the-loop） --
  async function runGold() {
    out.innerHTML = '<div class="loading">分層抽樣中…</div>';
    const r = await api.post("/api/gold-sample", { n: 10, stratify: true });
    out.innerHTML = "";
    const rows = r.rows || [];
    const inputs = [];
    const tbl = el("table"); tbl.append(el("tr", {}, ["段落（節選）", "算法預測", "人工：條文號（無=0）", "人工：模式"].map(h => el("th", {}, h))));
    rows.forEach(row => {
      const cid = el("input", { type: "text", style: "width:110px", placeholder: row.algo_clause_id });
      const md2 = el("input", { type: "text", style: "width:70px", placeholder: row.algo_mode });
      inputs.push([row, cid, md2]);
      tbl.append(el("tr", {}, [el("td", { class: "small" }, row.paragraph.slice(0, 60) + "…"), el("td", { class: "small" }, row.algo_clause_id + " / " + row.algo_mode), el("td", {}, cid), el("td", {}, md2)]));
    });
    const evalBtn = el("button", { class: "btn", onclick: async () => {
      const annotated = inputs.map(([row, cid, md2]) => ({ ...row, human_clause_id: cid.value.trim(), human_mode: md2.value.trim() }));
      const ev = await api.post("/api/gold-eval", { rows: annotated });
      out.prepend(el("div", { class: "cite-banner " + (ev.error ? "cite-warn" : "cite-ok") },
        ev.error ? ev.error : "條文級 P " + ev.clause_level.precision + " · R " + ev.clause_level.recall + " · F1 " + ev.clause_level.f1 + (ev.mode_agreement != null ? " · 模式一致 " + ev.mode_agreement : "")));
    } }, "計算 P/R/F1");
    out.append(el("div", { class: "card" }, [el("div", { class: "section-title" }, "標註 " + rows.length + " 段（" + r.n_strata + " 層分層抽樣）"), el("div", { class: "tbl-scroll" }, tbl), evalBtn, el("p", { class: "small muted" }, "確認算法正確：照抄預測值；否定：填正確條文號或 0。標註者判定是金標準。")]));
  }

  main.append(el("div", { class: "grid cols-2" }, [
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "笈成全庫（803 部 · 旁證層）"), el("div", { class: "row" }, [el("div", { style: "flex:1" }, libQ), el("button", { class: "btn", onclick: runLib }, "檢索")])]),
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "深度研究循環（七維溯源）"), el("div", { class: "row" }, [el("div", { style: "flex:1" }, drQ), el("button", { class: "btn", onclick: runDR }, "運行")])]),
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "評測看板"), el("button", { class: "btn ghost", onclick: runEval }, "載入四大基準")]),
    el("div", { class: "card" }, [el("div", { class: "section-title" }, "金標準標註閉環（human-in-the-loop）"), el("button", { class: "btn ghost", onclick: runGold }, "抽樣並標註")]),
  ]));
  main.append(out);
};

// ---------- router / boot ----------
async function boot() {
  try {
    const h = await api.get("/api/llm/status");
    META = { backend: h.backend, available: h.available };
    const bb = $("#backend-badge"); bb.textContent = "後端 " + h.backend; bb.classList.add(h.available ? "live" : "local");
    const s = await api.get("/api/stats");
    $("#rules-badge").textContent = s.initial_rules + " 規則 · " + s.skills + " Skill";
  } catch (e) {}
  $("#nav").addEventListener("click", e => {
    const b = e.target.closest("button[data-view]"); if (!b) return;
    [...$("#nav").children].forEach(x => x.classList.remove("active")); b.classList.add("active");
    go(b.dataset.view);
  });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-scrim").addEventListener("click", closeDrawer);
  $("#drawer-back").addEventListener("click", () => {
    const prev = drawerHistory.pop();
    if (prev != null) { currentClauseRef = null; openClause(prev, false); }
    updateBackBtn();
  });
  await applyLang();
  go("dashboard");
}
async function go(name) { const main = $("#main"); main.innerHTML = '<div class="loading">載入中…</div>'; try { await (views[name] || views.dashboard)(main); } catch (e) { main.innerHTML = '<div class="card">出錯：' + esc(e.message) + "</div>"; } }
boot();
