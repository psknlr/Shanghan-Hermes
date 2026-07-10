"""服務端可信 Principal 與角色策略（九輪評審 P0-1：角色不可由請求方自提權）。

原則：**角色是服務端身份的屬性，不是請求體參數**。

- 配置 ``HERMES_API_KEYS="token:role[:subject],..."`` 時：每個請求必須攜帶
  合法 token（Authorization: Bearer / X-Auth-Token）；角色上限由 key 綁定
  決定，請求體 role 只能同級或降級（patient < student < researcher <
  doctor），自提權返回 403 policy_denied 並記錄裁決。
- 僅配置 ``HERMES_SERVER_TOKEN``（舊接口，向後兼容）：單 token、全角色上限。
- 兩者皆未配置（本地開發模式）：匿名主體，上限由 ``HERMES_ANON_ROLE``
  控制（默認 doctor——本地單用戶自有進程，任何「限制」都是表演；公網部署
  必須配 HERMES_API_KEYS，或以 ``HERMES_ANON_ROLE=patient`` 把整個匿名面
  硬裁到患者安全層）。

誠實邊界：沒有身份就沒有真正的權限控制；本模塊保證的是「一旦有服務端
身份，能力上限就跟身份走、與請求體無關」，且每次拒絕裁決都可審計。
JWT/OIDC/反向代理映射屬部署層，見 docs/HARNESS.md 治理節。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ROLE_RANK = {"patient": 0, "student": 1, "researcher": 2, "doctor": 3}


class PolicyDenied(Exception):
    """請求方要求的能力超出其服務端身份授權。"""

    def __init__(self, reason: str, requested: str = "", ceiling: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.requested = requested
        self.ceiling = ceiling


@dataclass(frozen=True)
class PrincipalContext:
    """服務端解析出的可信主體（請求體無法偽造）。"""
    subject_id: str
    role_ceiling: str                  # 服務端綁定的最高角色
    auth_level: str                    # none | legacy_token | api_key
    tenant_id: str = "local"

    @property
    def rank(self) -> int:
        return ROLE_RANK.get(self.role_ceiling, 0)


def parse_api_keys(raw: str) -> Dict[str, Tuple[str, str]]:
    """``token:role[:subject]`` 逗號分隔 → {token: (role, subject)}。
    非法 role 的條目直接丟棄（fail-closed：寧可拒絕不可默認全權）。"""
    out: Dict[str, Tuple[str, str]] = {}
    for item in (raw or "").split(","):
        parts = item.strip().split(":")
        if len(parts) < 2 or not parts[0]:
            continue
        token, role = parts[0], parts[1].strip()
        if role not in ROLE_RANK:
            continue
        subject = parts[2].strip() if len(parts) > 2 and parts[2].strip() \
            else f"key-{role}"
        out[token] = (role, subject)
    return out


def resolve_principal(supplied_token: str, api_keys: Dict[str, Tuple[str, str]],
                      legacy_token: str) -> Optional[PrincipalContext]:
    """由服務端配置 + 請求憑證解析主體。返回 None = 401（憑證缺失/非法）。"""
    if api_keys:
        bound = api_keys.get(supplied_token or "")
        if bound is None:
            return None
        role, subject = bound
        return PrincipalContext(subject_id=subject, role_ceiling=role,
                                auth_level="api_key")
    if legacy_token:
        if supplied_token != legacy_token:
            return None
        return PrincipalContext(subject_id="legacy-token",
                                role_ceiling="doctor",
                                auth_level="legacy_token")
    anon_role = os.environ.get("HERMES_ANON_ROLE", "doctor")
    if anon_role not in ROLE_RANK:
        anon_role = "patient"          # 配錯值時收斂到最小權限
    return PrincipalContext(subject_id="anonymous", role_ceiling=anon_role,
                            auth_level="none")


def effective_role(principal: PrincipalContext,
                   requested: Optional[str]) -> Optional[str]:
    """裁定本請求的生效角色。

    - requested 為空：doctor 上限主體返回 None（交由智能體按問題推斷——
      推斷結果也在上限之內）；受限主體直接鉗到其上限角色。
    - requested 超出上限：拋 PolicyDenied（顯式 403，可審計；不做靜默降級）。
    """
    if not requested:
        return None if principal.rank >= ROLE_RANK["doctor"] \
            else principal.role_ceiling
    if requested not in ROLE_RANK:
        raise PolicyDenied(f"未知角色 {requested!r}", requested=str(requested),
                           ceiling=principal.role_ceiling)
    if ROLE_RANK[requested] > principal.rank:
        raise PolicyDenied(
            f"角色自提權被拒：請求 {requested}，服務端身份上限 "
            f"{principal.role_ceiling}（角色由身份綁定，不由請求體聲明）",
            requested=requested, ceiling=principal.role_ceiling)
    return requested


def allow_min_role(principal: PrincipalContext, min_role: str) -> bool:
    """端點級能力矩陣：主體上限低於端點最低角色即拒絕。"""
    return principal.rank >= ROLE_RANK.get(min_role, 0)


@dataclass(frozen=True)
class RequestContext:
    """不可變請求上下文（十一輪 P0-2）：所有路由顯式接收，業務代碼
    **禁止**自行從 body/query 取 role 或設默認角色——生效角色只此一處。
    effective_role=None 僅出現在 doctor 上限主體未聲明角色時（交由智能體
    按問題推斷，推斷結果仍在上限之內）。"""
    principal_id: str
    tenant_id: str
    role_ceiling: str
    effective_role: Optional[str]
    request_id: str
    purpose_of_use: str = "classical_text_research"

    def role_or(self, default: str) -> str:
        """受限主體永遠拿到自己的生效角色；只有全權主體未聲明時才用
        路由默認值（默認值不可能高於上限——上限即 doctor）。"""
        return self.effective_role or default


# 患者端序列化出口投影：無論業務函數是否記得脫敏，這些字段一律移除
# （處方結構/組成劑量/煎服法/加減方案——可執行診療信息不出患者面）
PATIENT_FORBIDDEN_KEYS = frozenset({
    "formula_blocks", "composition", "administration_notes", "dose_ratios",
    "dose", "family_dose_evolution", "modification_relations",
    "matched_formula_patterns", "hypotheses", "rescue_formulas",
})


def project_for_role(payload, role: Optional[str]):
    """角色投影（序列化出口，Business result → Role Projection →
    Response）。目前只有 patient 有強制刪除集；其他角色原樣通過。"""
    if role != "patient" or not isinstance(payload, (dict, list)):
        return payload
    removed: list = []

    def _strip(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in PATIENT_FORBIDDEN_KEYS:
                    removed.append(k)
                    continue
                out[k] = _strip(v)
            return out
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    projected = _strip(payload)
    if isinstance(projected, dict) and removed:
        projected["_role_projection"] = {
            "role": "patient",
            "removed_fields": sorted(set(removed)),
            "note": "患者端序列化出口投影：處方結構/劑量/煎服法等可執行"
                    "診療信息已強制移除（不依賴業務函數自行脫敏）"}
    return projected
