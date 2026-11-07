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
# 占位被剔除后留下的空格会顶在中文标点前面
_SPACE_BEFORE_CJK_RE = re.compile(r"\s+(?=[。，、；：？！）】」』])")

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


# 账本条目的固定结构：- [标签] 触发锚点=xxx；任务=yyy；结果=zzz
_LEDGER_TASK_MARK = "；任务="
_LEDGER_RESULT_MARK = "；结果="
_PENDING_PREFIX = "本轮待处理："
_PENDING_SUFFIX = "。这是当前事件对应的工作，可以继续决策。"


def strip_ledger_images(ledger_text: str) -> str:
    """剔除账本文本里的图片占位。

    任务只剩图片的条目整条去掉——图片被忽视时，这类条目对判定没有意义。
    一条不剩时返回空串，调用方就不该再把账本塞进上下文。
    """
    text = (ledger_text or "").strip()
    if not text:
        return ""

    kept: List[str] = []
    entries = 0
    for raw in text.splitlines():
        line = _SPACE_BEFORE_CJK_RE.sub("", _IMAGE_PLACEHOLDER_RE.sub(" ", raw)).strip()
        if not line:
            continue

        task = _ledger_task(line)
        if task is not None:
            if not task.strip():
                continue
            entries += 1
        elif line.startswith(_PENDING_PREFIX):
            pending = line[len(_PENDING_PREFIX) :]
            if pending.endswith(_PENDING_SUFFIX):
                pending = pending[: -len(_PENDING_SUFFIX)]
            if not pending.strip():
                continue

        kept.append(line)

    if entries == 0:
        return ""
    return "\n".join(kept)


def _ledger_task(line: str) -> Optional[str]:
    """账本条目行里取任务文本；不是条目行返回 None。"""
    if not line.startswith("- ["):
        return None
    marker = line.find(_LEDGER_TASK_MARK)
    if marker < 0:
        return None
    task = line[marker + len(_LEDGER_TASK_MARK) :]
    result_mark = task.find(_LEDGER_RESULT_MARK)
    if result_mark >= 0:
        task = task[:result_mark]
    return task


def split_by_mode(
    messages: Optional[List[Dict[str, Any]]],
    mode: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """按模式挑出要处理的消息，并标出当前消息是否纯图片。"""
    items = [m for m in (messages or []) if isinstance(m, dict)]
    current = items[-1] if items else None
    skip = mode == MODE_IGNORE and bool(current) and is_image_only(current)
    return items, skip
