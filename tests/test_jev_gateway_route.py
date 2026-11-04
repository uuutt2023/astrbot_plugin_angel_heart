"""客户端只跟网关打交道：发请求、读答案。"""

import json

import pytest
import pytest_asyncio
from aiohttp import web

QUESTION = {"should_reply": {"type": "noul", "instructions": "Should we reply?"}}


@pytest_asyncio.fixture
async def fake_gateway():
    """假网关：记录收到的请求，回一个 SystemOne 形状的答案。"""
    received = {}
    state = {"status": 200}

    async def handle(request: web.Request) -> web.Response:
        received["path"] = request.path
        received["auth"] = request.headers.get("authorization")
        received["token"] = request.headers.get("x-jev-gateway-token")
        received["idem"] = request.headers.get("idempotency-key")
        received["client"] = request.headers.get("x-jev-gateway-client")
        body = await request.read()
        received["body"] = json.loads(body) if body else {}
        if state["status"] != 200:
            return web.json_response({"error": "boom"}, status=state["status"])
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
    received["state"] = state
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
            {"proxy_url": fake_gateway["base"], "proxy_protocol": "systemone"},
            "对话内容",
            QUESTION,
        )
    finally:
        await client.close()

    assert result["status"] == "ready", result
    assert fake_gateway["path"] == "/jev/systemone"
    assert fake_gateway["auth"] is None
    assert fake_gateway["idem"]


@pytest.mark.asyncio
async def test_gateway_request_keeps_internal_shape(fake_gateway):
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
    assert body["state"] == "对话内容"
    assert body["questions"]["should_reply"]["type"] == "noul"
    # 模型由网关入口决定，客户端不带
    assert "model" not in body


@pytest.mark.asyncio
async def test_gateway_error_maps_to_status(fake_gateway):
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    fake_gateway["state"]["status"] = 402
    client = JevClient()
    try:
        result = await client.evaluate({"proxy_url": fake_gateway["base"]}, "x", QUESTION)
    finally:
        await client.close()
    assert result["status"] == "insufficient_credits"
    assert result["retryable"] is False


@pytest.mark.asyncio
async def test_unknown_protocol_maps_to_status(fake_gateway):
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    fake_gateway["state"]["status"] = 404
    client = JevClient()
    try:
        result = await client.evaluate({"proxy_url": fake_gateway["base"]}, "x", QUESTION)
    finally:
        await client.close()
    assert result["status"] == "protocol_not_found"


@pytest.mark.asyncio
async def test_missing_gateway_is_unconfigured():
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    try:
        result = await client.evaluate({}, "x", QUESTION)
    finally:
        await client.close()
    assert result["status"] == "unconfigured"


@pytest.mark.asyncio
async def test_idempotency_key_is_stable_within_one_call(fake_gateway):
    """同一次逻辑调用的重试要复用同一把键，才不会重复计费。"""
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient, new_idempotency_key

    key = new_idempotency_key()
    client = JevClient()
    try:
        await client.evaluate({"proxy_url": fake_gateway["base"]}, "x", QUESTION, idem=key)
        first = fake_gateway["idem"]
        await client.evaluate({"proxy_url": fake_gateway["base"]}, "x", QUESTION, idem=key)
    finally:
        await client.close()
    assert first == key
    assert fake_gateway["idem"] == key


class TestGatewayTokenFromUrl:
    """Docker 跨容器时令牌跟着地址走，客户端仍然只填两项配置。"""

    def test_resolve_token(self):
        from astrbot_plugin_angel_heart.core.jev_transport import resolve_token

        assert resolve_token("http://secret@astrbot:8770") == "secret"
        assert resolve_token("http://astrbot:8770") == ""
        assert resolve_token("") == ""

    def test_strip_userinfo(self):
        from astrbot_plugin_angel_heart.core.jev_transport import strip_userinfo

        assert strip_userinfo("http://secret@astrbot:8770") == "http://astrbot:8770"
        assert strip_userinfo("http://astrbot:8770") == "http://astrbot:8770"

    @pytest.mark.asyncio
    async def test_token_is_sent_as_header(self, fake_gateway):
        from astrbot_plugin_angel_heart.core.jev_transport import JevClient

        host = fake_gateway["base"].split("//", 1)[1]
        client = JevClient()
        try:
            await client.evaluate({"proxy_url": f"http://secret@{host}"}, "x", QUESTION)
        finally:
            await client.close()
        assert fake_gateway["token"] == "secret"
        # 客户端不带自己的鉴权，鉴权是网关换的
        assert fake_gateway["auth"] is None

    @pytest.mark.asyncio
    async def test_no_token_header_without_userinfo(self, fake_gateway):
        from astrbot_plugin_angel_heart.core.jev_transport import JevClient

        client = JevClient()
        try:
            await client.evaluate({"proxy_url": fake_gateway["base"]}, "x", QUESTION)
        finally:
            await client.close()
        assert fake_gateway["token"] is None


@pytest.mark.asyncio
async def test_request_carries_client_name(fake_gateway):
    """客户端自报家门，网关据此在拓扑图里显示这个插件。"""
    from astrbot_plugin_angel_heart.core.jev_transport import JevClient

    client = JevClient()
    cfg = {"proxy_url": fake_gateway["base"], "proxy_protocol": "systemone"}
    result = await client.evaluate(cfg, "hi", QUESTION)
    assert result["status"] == "ready"
    assert fake_gateway["client"] == "astrbot_plugin_angel_heart"
    await client.close()
