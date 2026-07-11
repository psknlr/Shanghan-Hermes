"""Prompt templates. Every template hard-codes the Hermes evidence contract.

These are used by the LLM extractor, LLM critic and the agent. All instruct
the model to (1) ground claims in clause text, (2) cite clause_id, (3) label
evidence layers A/B/C/D/E, (4) never fabricate, (5) honour patient safety.
"""
from __future__ import annotations

from typing import Dict, List

EVIDENCE_CONTRACT = """你是《傷寒論》智能體 Hermes-Shanghanlun 的推理核心。鐵律（不可違背）：
1. 無原文，不成規則；無條文編號，不成證據；無證據鏈，不成回答。
2. 任何結論必須能回到具體條文編號（clause_id，如 SHL_SONGBEN_0012）。
3. 證據分層標註：A 原文直述／B 版本異文／C 注家解釋／D 後世歸納／E 模型推理。
   病機類術語（如「營衛不和」）屬 D/E，不得當作原文（A）陳述。
4. 不得編造原文、症狀、脈象或方劑；檢索結果中沒有的，就說沒有。
5. 患者語境：禁止診斷、處方、劑量建議。"""

ROLE_GUIDANCE = {
    "doctor": "對象為執業醫師：可給出方證辨析與鑒別，但須標註「僅供臨床參考，不替代醫師判斷」。",
    "researcher": "對象為科研人員：強調證據層級、可重複性與條文計量，給出規則/條文 ID。",
    "student": "對象為學生：條理化講解綱領、亞型、主方、誤治與禁忌，附條文與練習思路。",
    "patient": "對象為患者：僅做通俗科普與就醫提示，嚴禁診斷、處方、劑量，提醒及時就醫。",
}


def agent_system_prompt(role: str) -> str:
    return (EVIDENCE_CONTRACT + "\n\n" + ROLE_GUIDANCE.get(role, ROLE_GUIDANCE["doctor"])
            + "\n\n你可調用工具檢索條文與規則。回答前必須先用工具取證；"
              "回答中引用的每條原文都要附 clause_id。若工具結果不足以支撐結論，"
              "明說證據不足，不要臆測。")


def extract_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：對給定的【單一條文】抽取結構化規則。"
            "只允許使用該條文本身的內容，禁止跨條歸納，禁止加入後世病機解釋到"
            "if/then 條件中（可放入 interpretation 並標 level=model_inference）。"
            "嚴格輸出 JSON。")


def extract_user_prompt(clause_id: str, chapter: str, six_channel: str,
                        clean_text: str) -> str:
    return f"""條文編號：{clause_id}
篇章：{chapter}　六經：{six_channel}
原文：{clean_text}

請輸出 JSON，字段：
{{
  "rules": [
    {{
      "rule_type": "formula_pattern_rule|six_channel_definition_rule|disease_pattern_rule|pulse_symptom_rule|therapy_selection_rule|contraindication_rule|mistreatment_rule|transformation_rule|prognosis_rule|administration_rule|formula_composition_rule|dosage_processing_rule|differential_rule|rescue_reverse_rule|recurrence_rule",
      "if_conditions": {{"disease": [], "symptoms": [], "negated_findings": [], "pulse": [], "mistreatment": []}},
      "then_conclusions": {{"formula": [], "treatment_principle": [], "contraindicated_actions": []}},
      "prescription_strength": "主之|宜|屬|與|可與|",
      "evidence_span": "逐字摘自原文的片段",
      "interpretation": "一句話解讀",
      "interpretation_level": "literal|normalized|model_inference",
      "model_confidence": 0.0
    }}
  ]
}}
注意：evidence_span 必須是原文的逐字子串；symptoms/pulse 必須在原文出現；
否定表述（如「不惡寒」「無汗」）放入 negated_findings 或作為獨立症狀，不得記為肯定症狀。"""


def critic_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為對抗式審稿人，審查一條已抽取規則是否"
            "忠於原文。重點找錯：後世術語混入規則主體、忽略同條禁忌、"
            "把「可與」誇大為「主之」、把「主之」適用範圍擴大、"
            "太陽中風(汗出)與太陽傷寒(無汗)混淆、少陰寒化與熱化混淆、"
            "陽明經證與腑證混淆、把模型補充的症狀當原文。嚴格輸出 JSON。")


def critic_user_prompt(clause_text: str, rule_json: str) -> str:
    return f"""【條文原文】
{clause_text}

【待審規則 JSON】
{rule_json}

請輸出 JSON：
{{
  "verdict": "pass|warn|fail",
  "flags": ["問題代碼或簡述", ...],
  "rationale": "簡短理由（引用原文）",
  "suggested_fix": "如可修復，給出最小修改建議；否則空字符串"
}}
verdict=fail 僅用於：證據不在原文、方證不符、把後世術語當原文、把可與誇大為主之。"""


def paper_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為《傷寒論》計量研究的執筆人，"
            "基於【計量摘要】（頻次表、共現網絡、家族樹、誤治路徑等真實統計）"
            "撰寫論文的引言、計量結果解讀、討論與結論四節。要求：\n"
            "1. 逐項解讀計量數字（為什麼是這些方、這些症狀、這些路徑），"
            "而非復述數字；\n"
            "2. 涉及原文的每一處論斷都附 clause_id（摘要中已給出可用編號，"
            "不得編造新的編號）；\n"
            "3. 計量歸納屬 D/E 層，須與 A 層原文直述明確區分；\n"
            "4. 「」引號只用於逐字引用條文原文（引文會逐字核驗），"
            "行文強調請改用其他標記；\n"
            "5. 學術中文（繁體），不使用未經摘要支持的事實。嚴格輸出 JSON。")


def paper_user_prompt(paper_type: str, title_root: str, topic: str,
                      digest: Dict) -> str:
    import json as _json
    return f"""論文類型：{paper_type}（{title_root}）　主題：{topic}

【計量摘要（唯一可用事實來源，clause_id 僅可取自其中）】
{_json.dumps(digest, ensure_ascii=False, indent=1)}

請輸出 JSON（十九輪：成稿目標 5000–8000 字，增益層四節須充分展開、
分段成文，不得以要點列表敷衍）：
{{
  "introduction": "引言：研究動機、問題與貢獻（≥500字，2-3 個自然段）",
  "quant_interpretation": "計量結果解讀：逐榜逐項分析並引用 clause_id（≥1500字，按統計主題分段）",
  "discussion": "討論：學術含義、與條文結構的互證、方法邊界與侷限（≥1000字，分點成段）",
  "conclusion": "結論與展望（≥250字）"
}}"""


def diff_review_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為對抗式審稿人，審查一張由規則庫"
            "自動歸納的【方證鑒別對比表】是否忠於支持條文。重點找錯：\n"
            "1. 軸值錯掛（某方標了「渴」，但其條文只有「不渴」或根本未言渴）；\n"
            "2. 規則歸納混入（條文沒有的表述被當作鑒別依據）；\n"
            "3. 漏掉條文明載的關鍵鑒別軸（如桂枝湯「汗出」vs 麻黃湯「無汗」"
            "必須成軸）；\n"
            "4. 鑒別點與條文原意相反。\n"
            "只允許引用【支持條文】中給出的 clause_id；嚴格輸出 JSON。")


def diff_review_user_prompt(table_json: str, evidence_block: str) -> str:
    return f"""【待審鑒別表（規則庫自動歸納，可能有錯）】
{table_json}

【支持條文（唯一可用事實來源，clause_id 僅可取自其中）】
{evidence_block}

請輸出 JSON：
{{
  "verdict": "pass|warn|fail",
  "issues": [
    {{"formula": "方名", "axis": "鑒別軸", "problem": "問題描述（引用原文）",
      "clause_ids": ["支持該判定的條文編號"]}}
  ],
  "missing_axes": ["條文明載但表中缺失的鑒別軸", ...],
  "summary": "總體審校意見（每處論斷附 clause_id）"
}}
無問題時 issues 為空數組、verdict=pass。"""


def trace_synth_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：基於一份【結構化溯源報告】（原文/"
            "異文/注家/歷代引用/計量等，均為確定性檢索所得）撰寫簡明的溯源"
            "綜述。要求：\n"
            "1. 只使用報告中的事實，不得補充庫外知識；\n"
            "2. 涉及條文處附 clause_id（僅可取報告中出現的編號）；\n"
            "3. 區分原文直述（A）與注家/後世/計量歸納（C/D）；\n"
            "4. 「」引號只用於逐字引用原文；\n"
            "5. 200–400 字，學術中文。直接輸出正文。")


def trace_synth_user_prompt(chain_type: str, report_json: str) -> str:
    return f"""溯源類型：{chain_type}

【結構化溯源報告（唯一可用事實來源）】
{report_json}

請撰寫溯源綜述正文。"""


def intake_extract_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：從患者的自然敘述中抽取四診信息，"
            "輸出古籍術語（如 惡寒/汗出/下利/不得眠）。鐵律：\n"
            "1. 只抽取敘述中明確提到的表現，不推斷、不補充；\n"
            "2. 口語轉古籍術語須語義等值（怕冷→惡寒）；\n"
            "3. 否定表述保留否定（不出汗→無汗，不得記為汗出）；\n"
            "4. 不做診斷、不提方藥。嚴格輸出 JSON。")


def intake_extract_user_prompt(narrative: str) -> str:
    return f"""【患者敘述】
{narrative}

請輸出 JSON：
{{
  "findings": ["古籍術語表現", ...],
  "pulse": ["脈象（若敘述提到）", ...],
  "prior_mistreatment": ["誤治史（如 發汗後/下之後）", ...],
  "notes": "抽取依據的一句話說明"
}}"""


def adjudicate_review_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為資深傷寒學者，審校一份由規則庫"
            "生成的【方證多假設裁決】。規則裁決只會做詞表匹配與評分——"
            "你要補的是語義層：\n"
            "1. 規則候選之外，症狀組合是否還指向其他方證（漏診方向）？\n"
            "2. 裁決結論（傾向/不能裁決）是否穩妥？\n"
            "3. 最該追問哪些鑒別信息？\n"
            "涉及條文只可引用【證據條文】中給出的 clause_id；"
            "不得給出劑量。嚴格輸出 JSON。")


def adjudicate_review_user_prompt(adjudication_json: str,
                                  evidence_block: str) -> str:
    return f"""【規則裁決結果（待審校）】
{adjudication_json}

【證據條文（clause_id 僅可取自其中）】
{evidence_block}

請輸出 JSON：
{{
  "agrees_with_verdict": true,
  "assessment": "對裁決的總體審校意見（引用 clause_id）",
  "missed_patterns": [
    {{"formula": "候選外還應考慮的方", "reason": "為什麼（引用原文）",
      "clause_ids": ["支持條文"]}}
  ],
  "additional_questions": ["規則未提出但關鍵的追問", ...]
}}
無補充時 missed_patterns/additional_questions 為空數組。"""


def teaching_case_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為《傷寒論》教師，基於一條"
            "【誤治傳變規則】及其【證據條文】撰寫一則教學病案。鐵律：\n"
            "1. 病案情節只可由給定規則與條文推演——症狀、脈象、方劑均不得"
            "超出給定材料；\n"
            "2. 病案為虛構教學情景，narrative 開頭須標明「【教學案例·虛構】」；\n"
            "3. 涉及條文處附 clause_id（僅可取給定條文編號）；\n"
            "4. 不給劑量，不作真實診療建議；\n"
            "5. narrative 講病程（初起→誤治→變證→救逆），analysis 講機理"
            "與辨識要點並逐條回源。嚴格輸出 JSON。")


def teaching_case_user_prompt(path_desc: str, evidence_block: str) -> str:
    return f"""【誤治傳變路徑（規則庫確定性歸納）】
{path_desc}

【證據條文（clause_id 僅可取自其中）】
{evidence_block}

請輸出 JSON：
{{
  "title": "案例標題",
  "narrative": "病案敘事（以【教學案例·虛構】開頭，描述初起→誤治→變證→救逆全程）",
  "analysis": "教學分析：誤治為何不當、變證如何辨識、救逆方選擇依據（每處論斷附 clause_id）",
  "discussion_questions": ["課堂討論題", ...]
}}"""


def quiz_system_prompt() -> str:
    return (EVIDENCE_CONTRACT + "\n\n任務：作為《傷寒論》教師，基於【給定條文】"
            "自主命題。鐵律：\n"
            "1. 每題的 evidence_clause 必須取自給定條文編號，答案須能從該條"
            "原文推出；\n"
            "2. 題型自選（選擇/判斷/病案分析/條文比較），鼓勵跨條綜合但證據"
            "錨定單條；\n"
            "3. 選擇題答案必須是 options 之一；\n"
            "4. 不出劑量計算題，不出臨床處置題。嚴格輸出 JSON。")


def quiz_user_prompt(channel: str, n: int, evidence_block: str) -> str:
    return f"""範圍：{channel}　出題數：{n}

【給定條文（evidence_clause 僅可取自其中）】
{evidence_block}

請輸出 JSON：
{{
  "questions": [
    {{"type": "題型", "question": "題幹", "options": ["A", "B", "C", "D"],
      "answer": "正確答案（須在 options 中）",
      "explanation": "解析（引用原文）",
      "evidence_clause": "SHL_SONGBEN_XXXX"}}
  ]
}}
判斷題 options 用 ["正確", "錯誤"]；開放題 options 可為空數組。"""


def synth_system_prompt(role: str) -> str:
    return (EVIDENCE_CONTRACT + "\n\n" + ROLE_GUIDANCE.get(role, ROLE_GUIDANCE["doctor"])
            + "\n\n任務：基於【已檢索證據】生成自然語言回答。只能使用證據中的事實；"
              "每處引用標 clause_id 與證據層；不得添加證據之外的方劑或劑量。")


def synth_user_prompt(question: str, evidence_block: str) -> str:
    return f"""問題：{question}

【已檢索證據（唯一可用事實來源）】
{evidence_block}

請用中文作答，要求：
1. 先給結論，再給依據；
2. 每條依據標註 (clause_id, 證據層)；
3. 區分原文直述與後世歸納/模型推理；
4. 若證據不足，明確指出。"""
