"""网关中继：客户端把请求交给 Jev 网关。"""

import json

import pytest
import pytest_asyncio
from aiohttp import web

QUESTION = {"should_reply": {"type": "noul", "instructions": "Should we reply?"}}


@pytest_asyncio.fixture
async def fake_gateway():
    """假网关：记录收到的请求，回一个 SystemOne 形状的答案。"""
    received = {}

    async def handle(request: web.Request) -> web.Response:
        received["path"] = request.path
        received["auth"] = request.headers.get("authorization")
        received["idem"] = request.headers.get("idempotency-key")
        body = await request.read()
        received["body"] = json.loads(body) if body else {}
        return web.json_response(
            {
                "model": "jev-latest",
                "answers": {
                    "should_reply": {"type": "noul", "noul": 0.9, "confidence": 0.9}
                },
            }
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    received["base"] = f"http://127.0.0.1:{port}"
    yield received
    await runner.cleanup()


def test_resolve_proxy_reads_config():
    from astrbot_plugin_angel_heart.core.jev_transport import resolve_proxy

    assert resolve_proxy({"proxy_url": "http://127.0.0.1:8770/"}) == (
        "http://127.0.0.1:8770",
        "systemone",
    )
    assert resolve_proxy({"proxy_url": "http://x", "proxy_protocol": "custom"}) == (
        "http://x",
        "custom",
    )
    assert resolve_proxy({}) == ("", "")
    assert resolve_proxy(None) == ("", "")


@pytest.mark.asyncio
async def test_request_goes_through_gateway(fake_gateway):
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    try:
        result = await client.evaluate(
            {
                "proxy_url": fake_gateway["base"],
                "proxy_protocol": "systemone",
                "api_format": "systemone",
            },
            "对话内容",
            QUESTION,
        )
    finally:
        await client.close()

    assert result["status"] == "ready", result
    assert fake_gateway["path"] == "/jev/systemone"
    # 走网关时不带自己的鉴权
    assert fake_gateway["auth"] is None
    # 幂等键照旧带上
    assert fake_gateway["idem"]


@pytest.mark.asyncio
async def test_gateway_request_keeps_payload_shape(fake_gateway):
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    try:
        await client.evaluate(
            {"proxy_url": fake_gateway["base"], "proxy_protocol": "systemone"},
            "对话内容",
            QUESTION,
        )
    finally:
        await client.close()

    body = fake_gateway["body"]
    assert "state" in body and "questions" in body
    assert body["questions"]["should_reply"]["type"] == "noul"


@pytest.mark.asyncio
async def test_missing_key_is_fine_with_gateway(fake_gateway):
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    try:
        result = await client.evaluate(
            {"proxy_url": fake_gateway["base"]}, "x", QUESTION
        )
    finally:
        await client.close()
    assert result["status"] == "ready", result


@pytest.mark.asyncio
async def test_missing_key_without_gateway_is_unconfigured():
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    try:
        result = await client.evaluate({}, "x", QUESTION)
    finally:
        await client.close()
    assert result["status"] == "unconfigured"
