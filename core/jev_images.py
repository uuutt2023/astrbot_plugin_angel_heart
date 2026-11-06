"""Jev 判定前的图片处理。

三种模式：
- keep（默认）：保持现状，图片占位原样交给 Jev。
- ignore：图片不参与判断——上下文里的图片占位清掉，纯图片消息直接不回复。
- caption：先用插件的图片理解模型转述，再把转述结果交给 Jev。
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from astrbot.api import logger

MODE_KEEP = "keep"
MODE_IGNORE = "ignore"
MODE_CAPTION = "caption"

# 平台 outline 给出的通用图片占位（不带编号）
_IMAGE_PLACEHOLDER_RE = re.compile(r"(?:\s*\[图片\]\s*)+")

Captioner = Callable[[Dict[str, Any]], Awaitable[str]]


def resolve_image_mode(cfg: Optional[Dict[str, Any]]) -> str:
    """按配置解析图片模式：忽视优先，其次转述，最后原样。"""
    cfg = cfg or {}
    if bool(cfg.get("ignore_image_judgment", False)):
        return MODE_IGNORE
    if bool(cfg.get("transcribe_image_with_model", False)):
        return MODE_CAPTION
    return MODE_KEEP


def message_has_image(msg: Any) -> bool:
    """消息里是否带图片。"""
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") == "image_url" for item in content
    ):
        return True
    refs = msg.get("image_refs")
    return isinstance(refs, list) and any(
        isinstance(ref, str) and ref.strip() for ref in refs
    )


def message_text(msg: Any) -> str:
    """取消息正文，剥掉平台给的通用图片占位。"""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = ""
    return _IMAGE_PLACEHOLDER_RE.sub(" ", text).strip()


def is_image_only(msg: Any) -> bool:
    """消息是否只有图片、没有文字。"""
    return message_has_image(msg) and not message_text(msg)


def text_only_copy(msg: Dict[str, Any], text: str) -> Dict[str, Any]:
    """返回只带文字的消息副本：去掉图片组件与转述字段。"""
    new = dict(msg)
    new["content"] = text
    # image_caption 会让格式化器渲染 <图片附件>（含本地路径），对 Jev 没有价值
    new.pop("image_caption", None)
    new.pop("image_refs", None)
    return new


async def prepare_messages_for_jev(
    messages: Optional[List[Dict[str, Any]]],
    mode: str,
    captioner: Optional[Captioner] = None,
) -> List[Dict[str, Any]]:
    """按图片模式预处理交给 Jev 的消息，返回新列表（不改原消息）。"""
    if mode == MODE_KEEP:
        return list(messages or [])

    prepared: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or not message_has_image(msg):
            prepared.append(msg)
            continue

        text = message_text(msg)
        if mode == MODE_IGNORE:
            prepared.append(text_only_copy(msg, text))
            continue

        caption = await _caption_of(msg, captioner)
        if caption:
            text = f"{text} [图片: {caption}]".strip()
        else:
            text = f"{text} [图片无法读取]".strip()
        prepared.append(text_only_copy(msg, text))
    return prepared


async def _caption_of(msg: Dict[str, Any], captioner: Optional[Captioner]) -> str:
    """取图片转述，失败按空串处理（转述挂了不该拖垮判定）。"""
    if captioner is None:
        return ""
    try:
        return str(await captioner(msg) or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"AngelHeart[Jev]: 图片转述失败，按无转述处理：{type(e).__name__}")
        return ""


def describe_mode(mode: str) -> str:
    """给日志用的一句话描述。"""
    return {
        MODE_KEEP: "原样保留",
        MODE_IGNORE: "忽视图片",
        MODE_CAPTION: "转述后交给 Jev",
    }.get(mode, "原样保留")


def split_by_mode(
    messages: Optional[List[Dict[str, Any]]],
    mode: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """按模式挑出要处理的消息，并标出当前消息是否纯图片。"""
    items = [m for m in (messages or []) if isinstance(m, dict)]
    current = items[-1] if items else None
    skip = mode == MODE_IGNORE and bool(current) and is_image_only(current)
    return items, skip
