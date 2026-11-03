#!/usr/bin/env python3
"""Jev 网关连通性诊断：告诉你「500 到底是谁的问题」。

用法：

    python scripts/probe_jev.py --base-url https://api.example.com \
        --api-key sk-xxx --model your-model

脚本会按文档格式（POST {base_url}/v1/systemone）发两次最小请求：
一次 choice、一次 noul，然后把结果翻译成结论：

- ✅ 网关正常：能拿到答案；
- ❌ 401：Key 无效或没有这个网关的权限；
- ❌ 402：余额不足；
- ❌ 404：路径不对（这个网关不是 SystemOne 兼容的）；
- ❌ 400「not served by this plugin」：模型名对，但这个模型没有绑定到该端点的通道；
- ❌ 422：请求体被拒——按提示改载荷；
- ❌ 5xx：网关收到了请求、校验也过了，但**上游任务失败**——问题在网关/上游，不在请求体。

最后一种最容易被误判成「插件写错了」，所以单独用两种问题类型各试一次：
两种都 5xx，就基本可以确定不是自己的锅。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from astrbot_plugin_angel_heart.core.jev_transport import (  # noqa: E402
    JevClient,
    normalize_base_url,
)

CHOICE_QUESTION = {
    "q1": {
        "type": "choice",
        "instructions": "Which queue should own this ticket?",
        "criteria": {"billing": "Payments", "technical": "Product errors"},
    }
}
NOUL_QUESTION = {"q2": {"type": "noul", "instructions": "Is this urgent?"}}


def diagnose(status: str, http_status: int | None, error_type: str, message: str) -> tuple[str, str]:
    """把一次失败翻译成 (结论, 建议)。"""
    if status == "ready":
        return "✅ 网关正常", "已经能拿到答案，插件里填同样的 base_url / api_key / model 即可"

    if http_status == 401:
        return "❌ 认证失败", "Key 无效、被吊销，或不属于这个网关；确认复制的是这个网关的 Key"
    if http_status == 402:
        return "❌ 余额不足", "账户没有额度，去网关控制台充值"
    if http_status == 404:
        return "❌ 路径不对", (
            "这个网关没有 /v1/systemone 端点——它多半只是 OpenAI 兼容的中转站，"
            "不支持 Jev 的决策请求体"
        )
    if http_status == 422:
        return "❌ 请求体被拒", f"网关校验没通过：{message or error_type}"
    if http_status == 400 and "not served by this plugin" in (message or ""):
        return "❌ 模型没绑通道", (
            "模型名能被网关认出，但这个模型没有绑定到 SystemOne 端点的通道——"
            "换模型名，或问网关运营方开通"
        )
    if http_status == 400 and ("state is required" in (message or "")
                               or "questions must be" in (message or "")):
        return "❌ 请求体不完整", f"网关的校验器说了：{message}"
    if http_status is not None and http_status >= 500:
        return "❌ 上游任务失败", (
            f"网关收到了请求、校验也过了，是它后面的上游报错了（HTTP {http_status}："
            f"{message or '无详细信息'}）。这不是请求体的问题——换一份载荷也一样会 500。"
            "过一会儿重试，或者联系网关运营方"
        )
    return f"❌ 失败（{status}）", f"{message or error_type or '没有更多细节'}"


async def probe(cfg: dict, questions: dict, label: str) -> dict:
    # 每次探测用独立客户端：否则前一次失败的退避窗口会把后一次直接挡掉
    client = JevClient()
    try:
        result = await client.evaluate(
            cfg, "The customer has been billed twice and wants a refund today.", questions
        )
    finally:
        await client.close()
    status = str(result.get("status") or "error")
    http_status = result.get("http_status")
    error_type = str(result.get("error_type") or "")
    message = str(result.get("error_message") or "").strip()
    verdict, advice = diagnose(status, http_status, error_type, message)
    print(f"\n[{label}] HTTP {http_status or '-'}  status={status}")
    if message:
        print(f"  上游消息：{message}")
    print(f"  → {verdict}")
    print(f"    {advice}")
    return {
        "label": label,
        "status": status,
        "http_status": http_status,
        "message": message,
        "verdict": verdict,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Jev 网关连通性诊断")
    parser.add_argument("--base-url", required=True, help="网关地址，如 https://jevmodel.org")
    parser.add_argument("--api-key", default="", help="API Key（默认读 JEVMODEL_API_KEY）")
    parser.add_argument("--model", default="jev-latest", help="模型名")
    parser.add_argument("--timeout-sec", type=float, default=15.0, help="单次请求超时（秒）")
    parser.add_argument("--retries", type=int, default=0, help="失败重试次数（诊断时建议 0）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("JEVMODEL_API_KEY", "")
    if not api_key:
        print("需要 API Key：设置环境变量 JEVMODEL_API_KEY 或用 --api-key 传入")
        return 2

    url = normalize_base_url(args.base_url)
    print("=" * 68)
    print(f"目标：{url}")
    print(f"模型：{args.model}")
    print("=" * 68)

    cfg = {
        "api_key": api_key,
        "base_url": args.base_url,
        "model": args.model,
        "timeout_sec": args.timeout_sec,
        "retries": args.retries,
    }

    results = []
    results.append(await probe(cfg, CHOICE_QUESTION, "choice"))
    results.append(await probe(cfg, NOUL_QUESTION, "noul"))

    print()
    print("=" * 68)
    statuses = {r["status"] for r in results}
    http_statuses = {r["http_status"] for r in results}
    if statuses == {"ready"}:
        print("结论：网关和模型都没问题 ✅")
        return 0
    if http_statuses and all(s and s >= 500 for s in http_statuses):
        print("结论：两种问题类型都是 5xx —— 上游任务失败，不是你的请求体有问题 ❌")
        print("      可以把上面这段结论直接发给网关运营方。")
        return 1
    print("结论：网关/模型配置有问题，先按上面的建议逐项排查 ❌")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
