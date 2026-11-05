"""
AngelHeart 插件 - Jev 决策器

把 [Jev 决策模型](https://jevmodel.org/zh/docs/) 的类型化答案映射成秘书决策对象。

Jev 不生成文本，只回答类型化问题，所以这里的分工是：
- noul  → 是否回复 / 是否被追问 / 话题是否有价值（三个概率，各自比阈值）
- choice→ 回复策略（枚举选项，映射回中文展示）

发给决策模型的问题一律用英文（Jev 主训练语言是英文，CJK 精度较低），
选项 key 也用英文，解析边界再映射回中文，与 LLM 路径的展示保持一致。

传输层复用 oven_multi 的实现思路（core/jev_transport.py）：本地预检、幂等键、
错误码映射、单轮预算与退避重试。
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from ..models.analysis_result import SecretaryDecision
from .jev_transport import (
    DEFAULT_RETRIES,
    JevClient,
    budget_seconds,
    new_idempotency_key,
    probability,
    resolve_proxy,
)
from .utils import format_message_for_llm

# 策略枚举：线上 key 用英文，展示用中文。
STRATEGY_LABELS = {
    "answer_question": "回答提问",
    "join_topic": "加入话题",
    "offer_help": "提供帮助",
    "greet": "简单回应",
    "stay_silent": "保持沉默",
}

# 判定方式：sample＝按概率抽样（回复率随置信度平滑变化）；threshold＝超过阈值才接话。
REPLY_MODE_SAMPLE = "sample"
REPLY_MODE_THRESHOLD = "threshold"
REPLY_MODES = (REPLY_MODE_SAMPLE, REPLY_MODE_THRESHOLD)


def decide_should_reply(
    probability: Optional[float], threshold: float, mode: str = REPLY_MODE_SAMPLE
) -> tuple[bool, float]:
    """把「应当回复」的概率变成接话与否。

    两种口径：

    - ``sample``：抽一个 [0,1) 的随机数，落在概率以内就接话。
      回复率随置信度平滑变化——0.3 的消息有 30% 机会接话，0.9 的有 90%，
      没有阈值处的跳变。
    - ``threshold``：概率超过阈值才接话。行为可复现，但阈值附近是硬跳变
      （0.59 不接、0.61 就接）。

    Returns:
        tuple[bool, float]: (是否接话, 本次抽样值；threshold 口径恒为 0.0)
    """
    if probability is None:
        return False, 0.0
    if mode != REPLY_MODE_SAMPLE:
        return probability >= threshold, 0.0
    draw = random.random()
    return draw <= probability, draw


# 传输层失败状态 → 一句可读原因。
_STATUS_REASON = {
    "unauthorized": "API Key 无效或已被禁用",
    "forbidden": "账户未激活或无权访问",
    "payload_too_large": "请求体过大",
    "insufficient_credits": "Jev 账户余额不足",
    "conflict": "上一次请求仍在处理中",
    "invalid_request": "请求体未通过 Jev 校验",
    "invalid_answer": "Jev 返回的答案格式不合法",
    "rate_limited": "触发 Jev 限流",
    "upstream_error": "Jev 上游故障",
    "overloaded": "Jev 服务过载",
    "error": "Jev 请求失败",
    "closed": "Jev 客户端已关闭",
    "unconfigured": "未配置 API Key",
}


def describe_failure(result: Dict[str, Any]) -> str:
    """把传输层失败翻译成一句可读原因（带服务端给的 message）。"""
    status = str(result.get("status") or "error")
    parts = [_STATUS_REASON.get(status, status)]
    http_status = result.get("http_status")
    if http_status:
        parts.append(f"HTTP {http_status}")
    message = str(result.get("error_message") or "").strip()
    if message:
        parts.append(message)
    return " · ".join(parts)


#: 主判定题的默认提示词。用户在配置里填了「Jev 提示词」就用用户的。
DEFAULT_SHOULD_REPLY_INSTRUCTIONS = (
    "Should the assistant reply in this group chat right now? "
    "Judge with the conversation context. A reply is warranted when "
    "the assistant is addressed, asked for something, or can add real "
    "value without interrupting. Chatter between other members does "
    "not warrant a reply."
)


def build_questions(prompt: str = "") -> Dict[str, Dict[str, Any]]:
    """构造决策请求的问题集合（4 道，落在文档的 8 道上限内）。

    题型直接用 Jev 的 noul / choice 形态：noul 判「命题为真」的概率，
    choice 从固定选项里选一个。

    prompt 非空时替换主判定题（should_reply）的 instructions，其余判断保持内置。
    """
    instructions = (prompt or "").strip() or DEFAULT_SHOULD_REPLY_INSTRUCTIONS
    return {
        "should_reply": {
            "type": "noul",
            "instructions": instructions,
            "criteria": {
                "true": (
                    "A relevant, helpful reply to the current message is warranted: "
                    "the assistant is addressed or asked for something."
                ),
                "false": (
                    "The message is chatter between other members, needs no reply, "
                    "or interrupting would be intrusive."
                ),
            },
        },
        "is_questioned": {
            "type": "noul",
            "instructions": (
                "Is the current message directed at the assistant - a question, a "
                "request, a mention, or an invitation for it to continue? "
                "Judge with the conversation context."
            ),
            "criteria": {
                "true": (
                    "The message is addressed to the assistant or clearly invites "
                    "the assistant to answer."
                ),
                "false": (
                    "The message is aimed at other members or is self-expression; "
                    "the assistant is not being asked anything."
                ),
            },
        },
        "is_interesting": {
            "type": "noul",
            "instructions": (
                "Is the current topic worth joining for the assistant - does it fit "
                "the assistant's role, is there something useful to add, and is this "
                "an appropriate moment to speak up? "
                "Judge with the conversation context."
            ),
            "criteria": {
                "true": (
                    "The topic fits the assistant's role and there is something "
                    "useful to add right now."
                ),
                "false": (
                    "The topic does not fit the assistant's role, or speaking up "
                    "would be redundant or intrusive."
                ),
            },
        },
        "reply_strategy": {
            "type": "choice",
            "instructions": (
                "What should the assistant do next? Pick exactly one option and "
                "judge with the conversation context."
            ),
            "criteria": {
                "answer_question": "Answer the question that was asked, directly and helpfully.",
                "join_topic": "Join the ongoing topic with a relevant contribution.",
                "offer_help": "Offer concrete help, a suggestion, or a next step.",
                "greet": "Respond briefly to a greeting or a light remark.",
                "stay_silent": "Stay silent; nothing needs to be said right now.",
            },
        },
    }


def build_state(
    recent_dialogue: List[Dict],
    historical_context: Optional[List[Dict]] = None,
    alias: str = "AngelHeart",
    max_chars: int = 1200,
    history_rounds: int = 4,
) -> str:
    """把对话拼成发送给决策模型的 state。

    当前消息放在最前、更早的历史放在其后，超长裁剪时截掉的是旧历史而不是当前消息。
    """
    lines: List[str] = []
    for conv in list(recent_dialogue or []):
        if not isinstance(conv, dict) or not _has_content(conv):
            continue
        try:
            formatted = format_message_for_llm(conv, alias)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"AngelHeart[Jev]: 消息格式化失败，跳过一条：{type(e).__name__}")
            continue
        if formatted:
            lines.append(formatted)

    if not lines:
        return ""

    current = lines[-1]
    history = lines[:-1]
    for conv in list(historical_context or [])[-max(0, int(history_rounds)) :]:
        if not isinstance(conv, dict) or not _has_content(conv):
            continue
        try:
            formatted = format_message_for_llm(conv, alias)
        except Exception:  # noqa: BLE001
            continue
        if formatted:
            history.append(formatted)

    header = (
        "[Scene] Group chat with an AI assistant; messages oldest first. "
        "Current = message to judge; Context = earlier messages."
    )
    parts = [header, "[Current]", current]
    if history:
        parts.extend(["[Context]", "\n".join(history)])

    state = "\n".join(parts)
    if max_chars and max_chars > 0 and len(state) > max_chars:
        # 保留头部与当前消息，裁掉更早的历史
        keep = len(header) + len(current) + 64
        room = max(0, max_chars - keep)
        state = "\n".join(
            [header, "[Current]", current, "[Context]", history_text(history, room)]
        )
    return state


def history_text(history: List[str], room: int) -> str:
    """从最近一条历史往前取，直到填满给定长度。"""
    if room <= 0:
        return ""
    picked: List[str] = []
    size = 0
    for line in reversed(history):
        if size + len(line) > room:
            break
        picked.append(line)
        size += len(line)
    return "\n".join(reversed(picked))


def _has_content(conv: dict) -> bool:
    """判断消息是否带有可判断的正文。

    格式化器对没有正文的字典会输出 '[None]' 占位，把它塞给决策模型只是噪声。
    """
    content = conv.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    return True


def guess_topic(recent_dialogue: List[Dict], max_chars: int = 24) -> str:
    """决策模型不产出自由文本，用最新一条消息做话题近似。"""
    for conv in reversed(list(recent_dialogue or [])):
        if not isinstance(conv, dict):
            continue
        text = str(conv.get("content") or "").strip()
        if not text:
            continue
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    return ""


class JevDecider:
    """用 Jev 决策模型产出秘书决策。"""

    def __init__(self):
        self.client = JevClient()

    def status(self) -> Dict[str, Any]:
        """返回传输层的可观测状态（调用次数、缓存命中、余额等）。"""
        return self.client.status()

    async def close(self) -> None:
        """关闭底层 HTTP 会话。"""
        await self.client.close()

    async def decide(
        self,
        cfg: dict,
        recent_dialogue: List[Dict],
        historical_context: Optional[List[Dict]] = None,
        alias: str = "AngelHeart",
        work_ledger_text: str = "",
    ) -> Dict[str, Any]:
        """向 Jev 提问并返回决策原料。

        Returns:
            dict: ok 为 True 时含 probabilities / reply_strategy；失败时含 error。
        """
        cfg = cfg or {}
        if not resolve_proxy(cfg)[0]:
            return {"ok": False, "error": "未配置网关代理地址（jev_decision.proxy_url）"}

        timeout = budget_seconds(cfg)
        try:
            retries = max(0, min(3, int(cfg.get("retries", DEFAULT_RETRIES))))
        except (TypeError, ValueError):
            return {"ok": False, "error": "Jev 重试配置无效"}

        max_state_chars = _int(cfg.get("max_state_chars"), 1200)
        history_rounds = _int(cfg.get("history_rounds"), 4)
        state = build_state(
            recent_dialogue or [],
            historical_context,
            alias=alias,
            max_chars=max_state_chars,
            history_rounds=history_rounds,
        )
        if not state:
            return {"ok": False, "error": "没有可交给 Jev 判断的对话内容"}

        ledger = (work_ledger_text or "").strip()
        if ledger:
            state = f"{state}\n\n[Assistant work ledger]\n{ledger}"

        questions = build_questions(str(cfg.get("prompt") or ""))
        deadline = time.monotonic() + timeout
        result: Dict[str, Any] = {"status": "error"}
        # 一次逻辑调用一把幂等键：重试复用它才不会重复计费。
        idem = new_idempotency_key()
        for attempt in range(retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = await asyncio.wait_for(
                    self.client.evaluate(
                        dict(cfg, timeout_sec=timeout), state, questions, idem
                    ),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                return {"ok": False, "error": "Jev 本轮总预算超时"}
            if result.get("status") == "ready":
                return self._map_answers(result.get("answers") or {})
            if not result.get("retryable"):
                break
            if attempt >= retries:
                break
            # 服务端已经回过错误（5xx 按文档已退款），下一次用新键重跑：
            # 复用旧键会撞上「上一把键还在处理中」的 409。
            # 只有压根没收到响应（请求可能已被受理）时才沿用旧键，避免重复计费。
            if result.get("http_status"):
                idem = new_idempotency_key()
            wait = float(result.get("retry_after") or 0)
            if wait >= deadline - time.monotonic():
                break
            await asyncio.sleep(min(wait, max(0.0, deadline - time.monotonic())))

        return {"ok": False, "error": describe_failure(result)}

    @staticmethod
    def _map_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
        """把 Jev 的答案映射成决策原料（阈值判定由调用方按配置执行）。"""
        reply_node = answers.get("should_reply")
        questioned_node = answers.get("is_questioned")
        interesting_node = answers.get("is_interesting")
        strategy_node = answers.get("reply_strategy")
        choice = ""
        if isinstance(strategy_node, dict):
            choice = str(strategy_node.get("choice") or "")
        return {
            "ok": True,
            "probabilities": {
                "should_reply": probability(reply_node),
                "is_questioned": probability(questioned_node),
                "is_interesting": probability(interesting_node),
            },
            "reply_strategy": choice,
        }


def _int(value: Any, default: int) -> int:
    """宽松取整数，异常回退默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    """宽松取浮点数，异常回退默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_decision(
    raw: Dict[str, Any],
    recent_dialogue: List[Dict],
    cfg: dict,
    reply_even_not_questioned: bool = False,
) -> SecretaryDecision:
    """把概率与选项装配成 SecretaryDecision。

    与 LLM 路径保持同一套「正向条件」规则：没有正当理由就不回复，
    除非开启了「非提问也回复」。
    """
    probabilities = raw.get("probabilities") or {}
    reply_threshold = _float(cfg.get("reply_threshold"), 0.6)
    questioned_threshold = _float(cfg.get("questioned_threshold"), 0.5)
    interesting_threshold = _float(cfg.get("interesting_threshold"), 0.5)
    mode = str(cfg.get("reply_mode") or REPLY_MODE_SAMPLE).strip() or REPLY_MODE_SAMPLE
    if mode not in REPLY_MODES:
        mode = REPLY_MODE_SAMPLE

    p_reply = probabilities.get("should_reply")
    p_questioned = probabilities.get("is_questioned")
    p_interesting = probabilities.get("is_interesting")

    # 主判定走判定方式（抽样 / 阈值）；证据信号仍用阈值，
    # 它们回答的是「有没有正当理由」，不是「这次要不要开口」。
    should_reply, draw = decide_should_reply(p_reply, reply_threshold, mode)
    is_questioned = p_questioned is not None and p_questioned >= questioned_threshold
    is_interesting = p_interesting is not None and p_interesting >= interesting_threshold

    strategy_key = str(raw.get("reply_strategy") or "")
    strategy = STRATEGY_LABELS.get(strategy_key, "继续观察")
    if strategy_key == "stay_silent":
        should_reply = False

    decision = SecretaryDecision(
        should_reply=should_reply,
        is_questioned=is_questioned,
        is_interesting=is_interesting,
        reply_strategy=strategy if should_reply else "继续观察",
        topic=guess_topic(recent_dialogue),
        reply_target="",
        entities=[],
        facts=[],
        keywords=[],
        confidence=float(p_reply) if p_reply is not None else 0.0,
    )
    logger.debug(
        f"AngelHeart[Jev]: 判定方式={mode} 概率={p_reply} "
        f"抽样={round(draw, 4) if draw else 0.0} 阈值={reply_threshold} "
        f"→ 回复={decision.should_reply}"
    )

    # 正向条件：与 llm_analyzer 同一套规则
    if decision.should_reply and not (
        decision.is_questioned or decision.is_interesting or reply_even_not_questioned
    ):
        logger.info(
            "AngelHeart[Jev]: should_reply=true 但无正当理由（非提问、非兴趣、未开启无视条件），设为不回复"
        )
        decision.should_reply = False
        decision.reply_strategy = "继续观察"

    return decision
