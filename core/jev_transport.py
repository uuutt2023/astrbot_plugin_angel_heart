"""通过 Jev 决策网关请求决策。

插件侧只做两件事：把请求发给网关、把答案读出来。
协议翻译（SystemOne ⇄ Decisions）、重试、30 秒去重缓存、余额追踪全在网关那边，
新接插件只要填「网关代理地址 + 协议名」两项即可。
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

import aiohttp

DEFAULT_TIMEOUT_SEC = 8

#: 网关侧会自己重试，这里再兜一层
DEFAULT_RETRIES = 1

#: 网关返回的 HTTP 状态 → 可读状态（与网关侧同一套口径）
_STATUS_BY_HTTP = {
    400: "invalid_request",
    401: "unauthorized",
    402: "insufficient_credits",
    403: "forbidden",
    404: "protocol_not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    500: "upstream_error",
    502: "upstream_error",
    503: "overloaded",
    504: "timeout",
    529: "overloaded",
}


def budget_seconds(cfg) -> float:
    """等待网关的总预算（秒）。"""
    value = cfg.get("timeout_sec") if isinstance(cfg, dict) else None
    try:
        budget = float(value or DEFAULT_TIMEOUT_SEC)
    except (TypeError, ValueError):
        budget = DEFAULT_TIMEOUT_SEC
    return max(1.0, min(20.0, budget))


def probability(node):
    value = node.get("noul") if isinstance(node, dict) else None
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        else None
    )


def new_idempotency_key() -> str:
    """为一次逻辑调用生成幂等键。

    键必须区分「同一次调用的重试」与「另一次独立调用」：按内容派生会让两条
    内容相同的消息互相冲突（服务端返回 409）。网关把同一个键带到上游，
    重试才不会重复计费。
    """
    return f"jev-{uuid.uuid4().hex}"


def resolve_proxy(cfg) -> tuple[str, str]:
    """网关代理地址与协议名；没配网关返回 ("", "")。"""
    if not isinstance(cfg, dict):
        return "", ""
    base = str(cfg.get("proxy_url") or "").strip().rstrip("/")
    if not base:
        return "", ""
    protocol = str(cfg.get("proxy_protocol") or "").strip() or "systemone"
    return base, protocol


def resolve_token(proxy_url: str) -> str:
    """从网关地址里取访问令牌。

    格式 http://<token>@host:port —— 这样客户端仍然只填「网关地址 + 协议名」
    两项，令牌跟着地址走。Docker 跨容器时网关要绑 0.0.0.0，令牌就是唯一的
    防线。
    """
    url = str(proxy_url or "")
    if "@" not in url:
        return ""
    _, sep, rest = url.partition("://")
    if not sep:
        rest = url
    userinfo, _, _host = rest.rpartition("@")
    return userinfo.strip()


def strip_userinfo(proxy_url: str) -> str:
    """把地址里的 userinfo（令牌）去掉，剩下的才是真正要连的地址。"""
    url = str(proxy_url or "")
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if not rest:
        return url
    _, _, host = rest.rpartition("@")
    return f"{scheme}://{host}"


class JevClient:
    """只跟网关打交道的客户端。"""

    def __init__(self):
        self.session = None
        self.closed = False
        self.last = {"status": "idle"}
        self.metrics = dict.fromkeys(("requests", "errors"), 0)

    async def evaluate(self, cfg, state, questions, idem=None) -> dict:
        """问网关一次。

        返回 {"status": "ready", "answers": {...}} 或 {"status": "<原因>"}。
        """
        if self.closed:
            return {"status": "closed"}
        proxy_base, protocol = resolve_proxy(cfg)
        if not proxy_base:
            return {
                "status": "unconfigured",
                "retryable": False,
                "error_message": "未配置网关代理地址",
            }

        url = f"{strip_userinfo(proxy_base)}/jev/{protocol}"
        payload = {"state": state, "questions": questions}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        idem = idem or new_idempotency_key()
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idem,
        }
        token = resolve_token(proxy_base)
        if token:
            headers["X-Jev-Gateway-Token"] = token
        timeout = budget_seconds(cfg)

        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                json_serialize=lambda value: json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                )
            )
        self.metrics["requests"] += 1
        started = time.monotonic()
        try:
            async with self.session.post(
                url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=False,
            ) as response:
                raw = await response.read()
                if response.status == 200:
                    try:
                        data = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError):
                        data = None
                    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                        result = {
                            "status": "invalid_answer",
                            "retryable": False,
                            "error_message": "网关返回的响应里没有答案",
                        }
                    else:
                        result = {
                            "status": "ready",
                            "answers": data["answers"],
                            "model": str(data.get("model") or ""),
                            "retryable": False,
                        }
                else:
                    result = _error_result(response.status, raw)
        except aiohttp.ClientError as exc:
            self.metrics["errors"] += 1
            result = {
                "status": "error",
                "retryable": True,
                "error_message": (
                    f"连不上网关（{url}）：{exc}。"
                    "检查网关插件是否启用、监听端口是否一致；"
                    "Docker 跨容器时网关要绑 0.0.0.0，地址要用容器名而不是 127.0.0.1。"
                ),
            }
        except (asyncio.TimeoutError, TimeoutError):
            self.metrics["errors"] += 1
            result = {
                "status": "timeout",
                "retryable": True,
                "error_message": f"网关请求超时（{timeout} 秒）",
            }

        result["seconds"] = round(time.monotonic() - started, 3)
        self.last = {k: v for k, v in result.items() if k != "answers"}
        return result

    async def close(self) -> None:
        self.closed = True
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    def status(self):
        return {
            "running": not self.closed,
            "last": dict(self.last),
            "metrics": dict(self.metrics),
        }


def _error_result(status: int, raw: bytes) -> dict:
    """把网关的错误响应翻成状态 + 说明。"""
    message = ""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        message = str(data.get("error") or data.get("message") or "")
        problems = data.get("problems")
        if isinstance(problems, list) and problems:
            message = "；".join(str(p) for p in problems)
    return {
        "status": _STATUS_BY_HTTP.get(int(status), "error"),
        "retryable": int(status) in (409, 429, 500, 502, 503, 504, 529),
        "error_message": message or f"网关返回 HTTP {status}",
    }
