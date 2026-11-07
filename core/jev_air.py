"""Jev 判定前的读空气上下文。

参考 astrbot_plugin_group_chat_plus 的读空气思路：判断模型靠平铺的对话文本很难
稳定判断「这条是不是在跟我说话」「我是不是已经回太多了」，所以要把这些信号显式
写成结构化的上下文块喂给它。

块里的内容全部来自账本已有的消息，不额外调用模型。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .jev_images import MODE_IGNORE

# 明确表示「别再说了」的信号词
REFUSAL_PATTERNS = (
    "别烦我",
    "不想聊",
    "闭嘴",
    "滚",
    "走开",
    "烦死了",
    "够了",
    "别说了",
    "别理我",
    "不要理我",
    "别插嘴",
    "stop talking",
    "leave me alone",
)

# 连续回复轮次的疲劳分档：(轮次, 描述)
_FATIGUE_LEVELS = ((1, "light"), (3, "moderate"), (5, "heavy"))

_AT_RE = re.compile(r"@([^\s:：,，。！？!?]+)")


def _text_of(msg: Any) -> str:
    """取消息正文（不含图片占位）。"""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _is_assistant(msg: Any) -> bool:
    """消息是不是助理自己说的。"""
    return isinstance(msg, dict) and msg.get("role") == "assistant"


def current_sender(messages: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """取当前发言者（最新一条用户消息）。"""
    items = [m for m in (messages or []) if isinstance(m, dict)]
    for msg in reversed(items):
        if not _is_assistant(msg):
            return msg
    return {}


def at_targets(msg: Dict[str, Any]) -> List[str]:
    """消息里 @ 到的人名。"""
    names = []
    for match in _AT_RE.finditer(_text_of(msg)):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def trailing_assistant_rounds(messages: Optional[List[Dict[str, Any]]]) -> int:
    """末尾连续几轮是助理自己在回。"""
    count = 0
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, dict):
            break
        if _is_assistant(msg):
            count += 1
            continue
        break
    return count


def refusal_signals(
    recent_dialogue: Optional[List[Dict[str, Any]]],
    historical_context: Optional[List[Dict[str, Any]]],
) -> List[str]:
    """上下文里出现的拒绝信号。"""
    hits: List[str] = []
    for msg in list(historical_context or []) + list(recent_dialogue or []):
        if not isinstance(msg, dict) or _is_assistant(msg):
            continue
        text = _text_of(msg)
        for pattern in REFUSAL_PATTERNS:
            if pattern in text and pattern not in hits:
                hits.append(pattern)
    return hits


def assistant_replies(
    recent_dialogue: Optional[List[Dict[str, Any]]],
    historical_context: Optional[List[Dict[str, Any]]],
    limit: int = 3,
) -> List[str]:
    """助理自己最近说过的话（防重复用）。"""
    replies: List[str] = []
    for msg in list(historical_context or []) + list(recent_dialogue or []):
        if not isinstance(msg, dict) or not _is_assistant(msg):
            continue
        text = _text_of(msg).strip()
        if text:
            replies.append(text[:120])
    return replies[-limit:]


def _fatigue_note(rounds: int) -> str:
    """连续回复轮次的疲劳提示。"""
    level = "none"
    for threshold, name in _FATIGUE_LEVELS:
        if rounds >= threshold:
            level = name
    if level == "heavy":
        return (
            f"The assistant already replied {rounds} times in a row (heavy fatigue): "
            "reply only if the message is clearly important or explicitly asks for help."
        )
    if level == "moderate":
        return (
            f"The assistant already replied {rounds} times in a row (moderate fatigue): "
            "be more selective and reply only to meaningful messages."
        )
    if level == "light":
        return (
            f"The assistant already replied {rounds} time(s) in a row (light fatigue): "
            "reply normally, but the topic may be winding down."
        )
    return ""


def build_air_context(
    recent_dialogue: Optional[List[Dict[str, Any]]],
    historical_context: Optional[List[Dict[str, Any]]],
    alias: str = "AngelHeart",
    image_mode: str = "",
) -> str:
    """构造读空气上下文块（没有可用信号时返回空串）。

    块里只写模型判断不了的东西：当前是谁在说、这条是不是在对别人说、
    我是不是已经回太多了、最近有没有人说别理我。
    """
    sender = current_sender(recent_dialogue)
    if not sender:
        return ""

    lines: List[str] = []
    name = str(sender.get("sender_name") or "").strip()
    sender_id = str(sender.get("sender_id") or "").strip()
    role = str(sender.get("sender_role") or "").strip()
    who = f"{name} (ID: {sender_id})" if sender_id else (name or "unknown member")
    if role:
        who += f", role: {role}"
    lines.append(f"Current sender: {who}. Judge the current message from this person.")

    targets = at_targets(sender)
    if sender.get("is_at_self"):
        lines.append("The current message mentions the assistant directly.")
    elif targets:
        lines.append(
            "The current message mentions "
            + ", ".join(f"@{t}" for t in targets)
            + " - it may be aimed at another member, not the assistant."
        )
    else:
        lines.append(
            "The current message mentions nobody; the assistant is not explicitly addressed."
        )

    rounds = trailing_assistant_rounds(recent_dialogue)
    note = _fatigue_note(rounds)
    if note:
        lines.append(note)

    refusals = refusal_signals(recent_dialogue, historical_context)
    if refusals:
        lines.append(
            "The user recently signalled they want less talking ("
            + ", ".join(f'"{r}"' for r in refusals)
            + "); do not push the conversation."
        )

    replies = assistant_replies(recent_dialogue, historical_context)
    if image_mode == MODE_IGNORE:
        # 图片被忽视时，助理自己说过的话里的图片占位也是无用信息
        replies = [_IMAGE_PLACEHOLDER_RE.sub(" ", r).strip() for r in replies]
        replies = [r for r in replies if r]
    if replies:
        lines.append(
            "The assistant's own earlier replies (do not repeat them): "
            + " | ".join(replies)
        )

    return "[Air reading]\n" + "\n".join(lines)


def mark_assistant_reply(formatted: str, msg: Any) -> str:
    """给助理自己的历史回复加标记，方便模型区分谁说的。

    账本里助理的消息格式化后没有前缀，模型分不出哪句是自己说的，
    只能靠「是不是自己刚说过」来防重复。
    """
    if formatted and _is_assistant(msg):
        return f"[Assistant's own earlier reply] {formatted}"
    return formatted
