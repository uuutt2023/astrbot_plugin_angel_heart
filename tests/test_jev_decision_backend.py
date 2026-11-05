"""
钉死 Jev 决策后端：状态拼装、限额预检、概率映射、秘书调度分支与可视化配置。

覆盖：
- build_state：当前消息在前、历史在后、超长裁剪；
- build_questions：问题集合落在 Jev 文档的限额内（≤8 题、instructions ≤1800、criteria ≤2000）；
- build_decision：概率 → 布尔、正向条件规则、策略映射；
- JevDecider.decide：成功路径与失败降级；
- Secretary.perform_analysis：llm / jev / hybrid 三条分支；
- _conf_schema.json 与 web_api：决策方式可下拉选择且取值受 options 约束。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def make_msg(name: str, text: str, ts: float = 0.0) -> dict:
    """构造一条最小消息字典。"""
    return {
        "role": "user",
        "content": text,
        "sender_name": name,
        "sender_id": name,
        "timestamp": ts,
    }


def make_config(backend: str = "llm", jev_enabled: bool = True, **jev_overrides) -> "object":
    """构造带 Jev 配置的 ConfigManager。"""
    from astrbot_plugin_angel_heart.core.config_manager import ConfigManager

    jev = {
        "enabled": jev_enabled,
        "proxy_url": "http://127.0.0.1:8770",
        "proxy_protocol": "systemone",
        # 夹具固定用阈值口径：阈值判定是确定的，抽样口径要另行测分布
        "reply_mode": "threshold",
        "reply_threshold": 0.6,
        "questioned_threshold": 0.5,
        "interesting_threshold": 0.5,
        "max_state_chars": 1200,
        "history_rounds": 4,
    }
    jev.update(jev_overrides)
    return ConfigManager(
        {
            "analyzer_model": "mock-model",
            "decision_backend": backend,
            "jev_decision": jev,
            "wake_interaction": {
                "force_reply_when_summoned": True,
                "reply_even_not_questioned": True,
                "alias": "fairy",
            },
            "timing": {},
            "leave_reply": {},
            "access_control": {},
            "context_compression": {},
            "output_rewrite": {},
        }
    )


def ready_answers(reply=0.9, questioned=0.8, interesting=0.3, strategy="answer_question"):
    """构造一组 Jev 就绪答案。"""
    return {
        "should_reply": {"type": "noul", "noul": reply},
        "is_questioned": {"type": "noul", "noul": questioned},
        "is_interesting": {"type": "noul", "noul": interesting},
        "reply_strategy": {"type": "choice", "choice": strategy},
    }


def _run_in_app(coro_factory, payload=None):
    """在 Quart 应用上下文里跑 web_api 的协程（jsonify 需要 app context）。

    web_api 的 _ok/_err 返回 Quart Response，这里统一还原成 dict。
    """
    import asyncio
    import inspect

    from quart import Quart

    import astrbot_plugin_angel_heart.web_api as web_api_mod

    app = Quart(__name__)

    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    async def _as_dict(response):
        getter = getattr(response, "get_json", None)
        if getter is None:
            return response
        data = getter()
        if inspect.isawaitable(data):
            data = await data
        return data

    async def _main():
        if payload is not None:
            async def _get_json():
                return payload

            old = web_api_mod._get_json
            web_api_mod._get_json = _get_json
            try:
                return await _as_dict(await _maybe_await(coro_factory()))
            finally:
                web_api_mod._get_json = old
        return await _as_dict(await _maybe_await(coro_factory()))

    async def _wrapped():
        async with app.app_context():
            return await _main()

    return asyncio.run(_wrapped())


class TestBuildState:
    """state 拼装：当前消息在前，历史在后。"""

    def test_current_message_before_history(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state([make_msg("A", "新消息")], [make_msg("B", "旧消息")])
        assert state.index("新消息") < state.index("旧消息")

    def test_empty_dialogue_returns_empty(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        assert build_state([]) == ""

    def test_truncation_keeps_current_message(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        long_text = "X" * 3000
        state = build_state(
            [make_msg("A", "当前消息")],
            [make_msg("B", long_text)],
            max_chars=400,
        )
        assert "当前消息" in state
        assert len(state) <= 400

    def test_history_is_appended_when_provided(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state(
            [make_msg("A", "新消息")], [make_msg("B", "旧消息")], alias="fairy"
        )
        assert "新消息" in state and "旧消息" in state

    def test_history_rounds_limits_carried_history(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        history = [make_msg("H", f"历史{i}") for i in range(10)]
        state = build_state(
            [make_msg("A", "当前")], history, max_chars=5000, history_rounds=2
        )
        assert "历史9" in state
        assert "历史0" not in state

    def test_blank_message_is_skipped(self):
        """没有正文的消息不进 state（避免 '[None]' 占位噪声）。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        assert build_state([make_msg("A", "   ")], []) == ""


class TestQuestionsWithinDocumentedLimits:
    """问题集合必须落在 Jev 文档的校验上限内，否则会被 422 拒绝。"""

    def test_question_count_within_limit(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        questions = build_questions()
        assert len(questions) == 4
        assert len(questions) <= 8

    def test_instructions_and_criteria_within_limits(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        for name, question in build_questions().items():
            assert len(question["instructions"]) <= 1800, name
            assert question.get("type") in ("noul", "choice", "score")
            if "criteria" in question:
                assert len(json.dumps(question["criteria"], ensure_ascii=False)) <= 2000, name

    def test_question_names_are_identifiers(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        for name in build_questions():
            assert name.replace("_", "").isalnum()
            assert len(name) <= 64

    def test_instructions_are_english(self):
        """Jev 主训练语言是英文，问题指令不应混入中文。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        for name, question in build_questions().items():
            assert question["instructions"].isascii(), name

    def test_strategy_has_five_options(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        criteria = build_questions()["reply_strategy"]["criteria"]
        assert set(criteria) == {
            "answer_question",
            "join_topic",
            "offer_help",
            "greet",
            "stay_silent",
        }


class TestBuildDecision:
    """概率 → 决策对象的映射。"""

    def _raw(self, reply=0.9, questioned=0.8, interesting=0.2, strategy="answer_question"):
        return {
            "ok": True,
            "probabilities": {
                "should_reply": reply,
                "is_questioned": questioned,
                "is_interesting": interesting,
            },
            "reply_strategy": strategy,
        }

    def test_probabilities_map_to_booleans(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(reply=0.9, questioned=0.8),
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert decision.should_reply is True
        assert decision.is_questioned is True
        assert decision.is_interesting is False

    def test_threshold_is_respected(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(reply=0.5, questioned=0.9),
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert decision.should_reply is False

    def test_missing_probability_is_not_reply(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            {"ok": True, "probabilities": {}, "reply_strategy": "answer_question"},
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert decision.should_reply is False

    def test_positive_condition_required(self):
        """should_reply 为真但无正当理由时仍不回复。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(reply=0.9, questioned=0.1, interesting=0.1),
            [make_msg("A", "闲聊")],
            {"reply_mode": "threshold"},
        )
        assert decision.should_reply is False

    def test_reply_even_not_questioned_allows_reply(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(reply=0.9, questioned=0.1, interesting=0.1),
            [make_msg("A", "闲聊")],
            {"reply_mode": "threshold"},
            reply_even_not_questioned=True,
        )
        assert decision.should_reply is True

    def test_strategy_key_maps_to_chinese_label(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(strategy="join_topic"),
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert decision.reply_strategy == "加入话题"

    def test_stay_silent_forces_no_reply(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(strategy="stay_silent"),
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert decision.should_reply is False

    def test_topic_falls_back_to_latest_message(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(), [make_msg("A", "讲个笑话")], {"reply_mode": "threshold"}
        )
        assert decision.topic == "讲个笑话"

    def test_probability_parser_rejects_invalid(self):
        """概率必须是 0-1 的有限数；非法值视为缺失。"""
        from astrbot_plugin_angel_heart.core.jev_transport import probability

        assert probability({"noul": 0.5}) == 0.5
        assert probability({"noul": 0.0}) == 0.0
        assert probability({"noul": 1.5}) is None
        assert probability({"noul": "x"}) is None
        assert probability({}) is None
        assert probability(None) is None


class TestJevDecider:
    """JevDecider.decide 的成功与失败路径。"""

    @pytest.mark.asyncio
    async def test_missing_gateway_returns_error(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        result = await JevDecider().decide({}, [make_msg("A", "你好")])
        assert result["ok"] is False
        assert "proxy_url" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_dialogue_returns_error(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        result = await JevDecider().decide({"proxy_url": "http://127.0.0.1:8770"}, [])
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_ready_answer_is_mapped(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        decider.client = AsyncMock()
        decider.client.evaluate.return_value = {
            "status": "ready",
            "answers": ready_answers(),
        }
        result = await decider.decide(
            {"proxy_url": "http://127.0.0.1:8770"}, [make_msg("A", "在吗")], alias="fairy"
        )
        assert result["ok"] is True
        assert result["probabilities"]["should_reply"] == 0.9
        assert result["probabilities"]["is_interesting"] == 0.3
        assert result["reply_strategy"] == "answer_question"

    @pytest.mark.asyncio
    async def test_failure_is_reported_with_reason(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        decider.client = AsyncMock()
        decider.client.evaluate.return_value = {
            "status": "unauthorized",
            "http_status": 401,
            "retryable": False,
            "error_message": "Invalid or disabled API key.",
        }
        result = await decider.decide({"proxy_url": "http://127.0.0.1:8770"}, [make_msg("A", "在吗")])
        assert result["ok"] is False
        assert "API Key" in result["error"]

    @pytest.mark.asyncio
    async def test_retry_uses_new_key_after_server_error(self):
        """服务端已回错误时换新幂等键重跑；压根没响应时沿用旧键。"""
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        decider.client = AsyncMock()
        decider.client.evaluate.side_effect = [
            {
                "status": "upstream_error",
                "http_status": 502,
                "retryable": True,
                "retry_after": 0,
            },
            {"status": "ready", "answers": ready_answers()},
        ]
        result = await decider.decide(
            {"proxy_url": "http://127.0.0.1:8770", "retries": 1}, [make_msg("A", "在吗")]
        )
        assert result["ok"] is True
        first_key = decider.client.evaluate.call_args_list[0][0][3]
        second_key = decider.client.evaluate.call_args_list[1][0][3]
        assert first_key != second_key

    @pytest.mark.asyncio
    async def test_no_retry_when_not_retryable(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        decider.client = AsyncMock()
        decider.client.evaluate.return_value = {
            "status": "unauthorized",
            "http_status": 401,
            "retryable": False,
        }
        result = await decider.decide(
            {"proxy_url": "http://127.0.0.1:8770", "retries": 2}, [make_msg("A", "在吗")]
        )
        assert result["ok"] is False
        assert decider.client.evaluate.await_count == 1


class TestSecretaryBackendDispatch:
    """秘书按 decision_backend 走 llm / jev / hybrid 三条分支。"""

    def _secretary(self, config):
        from astrbot_plugin_angel_heart.roles.secretary import Secretary

        angel_context = None
        secretary = Secretary(config, context=None, angel_context=angel_context)
        return secretary

    @pytest.mark.asyncio
    async def test_llm_backend_calls_analyzer(self):
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="llm"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )
        secretary.jev_decider.decide = AsyncMock()

        decision = await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert decision.should_reply is True
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 1
        assert secretary.jev_decider.decide.await_count == 0

    @pytest.mark.asyncio
    async def test_jev_backend_skips_analyzer(self):
        secretary = self._secretary(make_config(backend="jev"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock()
        secretary.jev_decider.decide = AsyncMock(
            return_value={
                "ok": True,
                "probabilities": {
                    "should_reply": 0.9,
                    "is_questioned": 0.8,
                    "is_interesting": 0.2,
                },
                "reply_strategy": "answer_question",
            }
        )

        decision = await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert decision.should_reply is True
        assert decision.reply_strategy == "回答提问"
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 0
        assert secretary.jev_decider.decide.await_count == 1

    @pytest.mark.asyncio
    async def test_jev_backend_disabled_falls_back_to_llm(self):
        """Jev 总开关关闭时不得走 Jev，仍用秘书模型。"""
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="jev", jev_enabled=False))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )
        secretary.jev_decider.decide = AsyncMock()

        await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 1
        assert secretary.jev_decider.decide.await_count == 0

    @pytest.mark.asyncio
    async def test_hybrid_gate_blocks_llm_call(self):
        """hybrid：Jev 门控未通过时不再调用秘书模型。"""
        secretary = self._secretary(make_config(backend="hybrid"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock()
        secretary.jev_decider.decide = AsyncMock(
            return_value={
                "ok": True,
                "probabilities": {
                    "should_reply": 0.1,
                    "is_questioned": 0.1,
                    "is_interesting": 0.1,
                },
                "reply_strategy": "stay_silent",
            }
        )

        decision = await secretary.perform_analysis([make_msg("A", "闲聊")], [], "chat")
        assert decision.should_reply is False
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 0

    @pytest.mark.asyncio
    async def test_hybrid_gate_passes_to_llm(self):
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="hybrid"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )
        secretary.jev_decider.decide = AsyncMock(
            return_value={
                "ok": True,
                "probabilities": {
                    "should_reply": 0.9,
                    "is_questioned": 0.8,
                    "is_interesting": 0.5,
                },
                "reply_strategy": "answer_question",
            }
        )

        await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_gate_failure_passes_to_llm(self):
        """门控自身失败不阻断主流程：放行给秘书模型。"""
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="hybrid"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )
        secretary.jev_decider.decide = AsyncMock(
            return_value={"ok": False, "error": "Jev 请求失败"}
        )

        await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 1

    @pytest.mark.asyncio
    async def test_jev_backend_failure_does_not_reply(self):
        """jev 模式下决策失败只告警，不回复。"""
        secretary = self._secretary(make_config(backend="jev"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock()
        secretary.jev_decider.decide = AsyncMock(
            return_value={"ok": False, "error": "Jev 请求失败"}
        )

        decision = await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert decision.should_reply is False
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 0


class TestConfigManagerBackend:
    """配置读取与回落。"""

    def test_default_backend_is_llm(self):
        assert make_config().decision_backend == "llm"

    def test_unknown_backend_falls_back(self):
        from astrbot_plugin_angel_heart.core.config_manager import ConfigManager

        cm = ConfigManager({"decision_backend": "nope"})
        assert cm.decision_backend == "llm"

    def test_known_backends(self):
        assert make_config(backend="jev").decision_backend == "jev"
        assert make_config(backend="hybrid").decision_backend == "hybrid"

    def test_jev_enabled_requires_gateway(self):
        """Jev 可用性 = 总开关打开 且 填了网关代理地址。"""
        assert make_config(jev_enabled=True).jev_decision_enabled is True
        assert make_config(jev_enabled=False).jev_decision_enabled is False
        assert make_config(jev_enabled=True, proxy_url="").jev_decision_enabled is False


class TestVisualConfigSelectable:
    """可视化配置必须能下拉选决策方式。"""

    def _schema(self) -> dict:
        return json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

    def test_schema_exposes_selectable_backend(self):
        meta = self._schema()["decision_backend"]
        assert meta["type"] == "string"
        assert meta["options"] == ["llm", "jev", "hybrid"]
        assert len(meta["labels"]) == 3
        # 副标题可选：要么没有，要么每个选项一条
        assert len(meta.get("subtitles", meta["options"])) == 3

    def test_schema_exposes_jev_group(self):
        """客户端只留「调用网关」与「协议名」两项连接配置。"""
        items = self._schema()["jev_decision"]["items"]
        for key in ("enabled", "proxy_url", "proxy_protocol"):
            assert key in items
        for gone in ("api_key", "base_url", "api_format", "endpoint_path", "model"):
            assert gone not in items

    def test_web_api_rejects_value_outside_options(self):
        import asyncio

        from astrbot_plugin_angel_heart.web_api import PluginConfigAPI

        api = PluginConfigAPI.__new__(PluginConfigAPI)
        api._config = MagicMock()
        api._config.__contains__.return_value = True
        api._schema = self._schema()

        result = _run_in_app(
            api.save_config, {"values": {"decision_backend": "bogus"}}
        )
        assert result["status"] == "error"

    def test_web_api_accepts_string_without_options(self):
        from astrbot_plugin_angel_heart.web_api import PluginConfigAPI

        api = PluginConfigAPI.__new__(PluginConfigAPI)
        api._config = MagicMock()
        api._schema = self._schema()
        api._plugin = MagicMock()

        result = _run_in_app(
            api.save_config, {"values": {"jev_decision": {"proxy_url": "http://127.0.0.1:8770"}}}
        )
        assert result["status"] == "ok"


class TestReplyModeSampling:
    """概率判定方式：抽样（平滑）vs 阈值（可复现）。"""

    def _raw(self, reply=0.9, questioned=0.8, interesting=0.3, strategy="answer_question"):
        return {
            "ok": True,
            "probabilities": {
                "should_reply": reply,
                "is_questioned": questioned,
                "is_interesting": interesting,
            },
            "reply_strategy": strategy,
        }

    def test_threshold_mode_is_deterministic(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        cfg = {"reply_mode": "threshold", "reply_threshold": 0.6}
        for _ in range(20):
            decision = build_decision(
                self._raw(reply=0.9), [make_msg("A", "你好")], cfg
            )
            assert decision.should_reply is True

        for _ in range(20):
            decision = build_decision(
                self._raw(reply=0.5), [make_msg("A", "你好")], cfg
            )
            assert decision.should_reply is False

    def test_sample_mode_follows_probability(self):
        """抽样口径下，概率越大越容易接话。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        def reply_rate(prob: float, n: int = 400) -> float:
            hits = 0
            for _ in range(n):
                decision = build_decision(
                    self._raw(reply=prob, questioned=1.0, interesting=1.0),
                    [make_msg("A", "你好")],
                    {"reply_mode": "sample"},
                )
                hits += 1 if decision.should_reply else 0
            return hits / n

        low = reply_rate(0.2)
        mid = reply_rate(0.5)
        high = reply_rate(0.9)
        assert low < mid < high, (low, mid, high)
        # 抽样是统计性的，留足余量
        assert abs(low - 0.2) < 0.08, low
        assert abs(high - 0.9) < 0.08, high

    def test_sample_mode_below_threshold_can_reply(self):
        """抽样口径下，低于阈值的概率也有机会接话（这是与 threshold 的区别）。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        replied = False
        for _ in range(200):
            decision = build_decision(
                self._raw(reply=0.3, questioned=1.0, interesting=1.0),
                [make_msg("A", "你好")],
                {"reply_mode": "sample", "reply_threshold": 0.6},
            )
            if decision.should_reply:
                replied = True
                break
        assert replied is True

    def test_evidence_gates_still_apply_in_sample_mode(self):
        """证据门（被追问/有兴趣）在抽样口径下照旧生效。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        for _ in range(100):
            decision = build_decision(
                self._raw(reply=1.0, questioned=0.1, interesting=0.1),
                [make_msg("A", "你好")],
                {"reply_mode": "sample"},
            )
            assert decision.should_reply is False

    def test_unknown_mode_falls_back_to_sample(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        replied = False
        for _ in range(200):
            decision = build_decision(
                self._raw(reply=0.4, questioned=1.0, interesting=1.0),
                [make_msg("A", "你好")],
                {"reply_mode": "nonsense"},
            )
            if decision.should_reply:
                replied = True
                break
        assert replied is True

    def test_decision_carries_confidence(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        decision = build_decision(
            self._raw(reply=0.11508464306322369),
            [make_msg("A", "你好")],
            {"reply_mode": "threshold"},
        )
        assert abs(decision.confidence - 0.11508464306322369) < 1e-12

    def test_stay_silent_overrides_sampling(self):
        """选了 stay_silent 就一定不接，抽样也不例外。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_decision

        for _ in range(200):
            decision = build_decision(
                self._raw(reply=1.0, strategy="stay_silent"),
                [make_msg("A", "你好")],
                {"reply_mode": "sample"},
            )
            assert decision.should_reply is False

    def test_hybrid_gate_uses_same_mode(self):
        """hybrid 门控与最终判定用同一套口径。"""
        from astrbot_plugin_angel_heart.core.jev_decider import decide_should_reply

        # threshold 口径：低于阈值一律不放行
        for _ in range(50):
            passed, _draw = decide_should_reply(0.3, 0.6, "threshold")
            assert passed is False
        # sample 口径：低概率也有机会放行
        hits = 0
        for _ in range(400):
            passed, _draw = decide_should_reply(0.3, 0.6, "sample")
            hits += 1 if passed else 0
        assert 0.15 < hits / 400 < 0.45

    def test_missing_probability_never_replies(self):
        from astrbot_plugin_angel_heart.core.jev_decider import decide_should_reply

        for mode in ("sample", "threshold"):
            for _ in range(50):
                passed, _draw = decide_should_reply(None, 0.6, mode)
                assert passed is False


class TestConfidenceScalesReplyLength:
    """概率缩放发言程度：把握越小，允许说的越少。"""

    def _front_desk(self):
        from astrbot_plugin_angel_heart.roles.front_desk import FrontDesk

        fd = FrontDesk.__new__(FrontDesk)
        fd._config_manager = MagicMock()
        cm = MagicMock()
        cm.normal_reply_max_chars = 20
        cm.focus_reply_max_chars = 200
        fd._config_manager.for_chat.return_value = cm
        return fd

    def test_no_confidence_keeps_original_limits(self):
        from astrbot_plugin_angel_heart.roles.front_desk import FrontDesk

        fd = self._front_desk()
        text = fd._build_reply_length_reminder(False, "chat", 0.0)
        assert "20 字" in text

    def test_low_confidence_shortens_limit(self):
        from astrbot_plugin_angel_heart.roles.front_desk import FrontDesk

        fd = self._front_desk()
        text = fd._build_reply_length_reminder(False, "chat", 0.2)
        assert "12 字" in text
        assert "把握不大" in text

    def test_high_confidence_keeps_near_full_limit(self):
        from astrbot_plugin_angel_heart.roles.front_desk import FrontDesk

        fd = self._front_desk()
        text = fd._build_reply_length_reminder(False, "chat", 1.0)
        assert "20 字" in text
        assert "把握不大" not in text

    def test_scale_helper_bounds(self):
        from astrbot_plugin_angel_heart.roles.front_desk import (
            _scale_limit_by_confidence,
        )

        assert _scale_limit_by_confidence(200, 0.0) == 200
        assert _scale_limit_by_confidence(200, 0.5) == 150
        assert _scale_limit_by_confidence(200, 1.0) == 200
        assert _scale_limit_by_confidence(200, 2.0) == 200
        assert _scale_limit_by_confidence(20, 0.2) == 12

    def test_confidence_read_from_event(self):
        import json as _json

        from astrbot_plugin_angel_heart.roles.front_desk import FrontDesk

        fd = self._front_desk()
        event = MagicMock()
        event.angelheart_context = _json.dumps(
            {"secretary_decision": {"should_reply": True, "confidence": 0.42}}
        )
        assert fd._decision_confidence(event) == 0.42

        event.angelheart_context = "not-json"
        assert fd._decision_confidence(event) == 0.0

        event.angelheart_context = ""
        assert fd._decision_confidence(event) == 0.0


class TestHybridGateConfidenceFlow:
    """hybrid 模式：Jev 的把握要跟着最终决策走。"""

    def _secretary(self, config):
        from astrbot_plugin_angel_heart.roles.secretary import Secretary

        return Secretary(config, context=None, angel_context=None)

    @pytest.mark.asyncio
    async def test_gate_confidence_attaches_to_llm_decision(self):
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="hybrid"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )
        secretary.jev_decider.decide = AsyncMock(
            return_value={
                "ok": True,
                "probabilities": {
                    "should_reply": 0.9,
                    "is_questioned": 0.8,
                    "is_interesting": 0.5,
                },
                "reply_strategy": "answer_question",
            }
        )

        decision = await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert decision.should_reply is True
        assert decision.confidence == 0.9

    @pytest.mark.asyncio
    async def test_llm_backend_keeps_confidence_zero(self):
        """纯 LLM 路径没有置信度概念，字数行为保持原样。"""
        from astrbot_plugin_angel_heart.models.analysis_result import SecretaryDecision

        secretary = self._secretary(make_config(backend="llm"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock(
            return_value=SecretaryDecision(
                should_reply=True, reply_strategy="策略", topic="话题",
                entities=[], facts=[], keywords=[]
            )
        )

        decision = await secretary.perform_analysis([make_msg("A", "在吗")], [], "chat")
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_gate_blocked_returns_no_reply(self):
        """门控未通过时直接不回复，且不调用秘书模型。"""
        secretary = self._secretary(make_config(backend="hybrid"))
        secretary.llm_analyzer.analyze_and_decide = AsyncMock()
        secretary.jev_decider.decide = AsyncMock(
            return_value={
                "ok": True,
                "probabilities": {
                    "should_reply": 0.1,
                    "is_questioned": 0.1,
                    "is_interesting": 0.1,
                },
                "reply_strategy": "stay_silent",
            }
        )

        decision = await secretary.perform_analysis([make_msg("A", "闲聊")], [], "chat")
        assert decision.should_reply is False
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 0


class TestLogProbabilityRounding:
    """日志里的概率保留 3 位小数。"""

    def test_round_probs(self):
        from astrbot_plugin_angel_heart.roles.secretary import _round_probs

        out = _round_probs(
            {"should_reply": 0.11508464306322369, "is_questioned": 0.8, "bad": None}
        )
        assert out["should_reply"] == 0.115
        assert out["is_questioned"] == 0.8
        assert out["bad"] is None

    def test_round_probs_handles_non_dict(self):
        from astrbot_plugin_angel_heart.roles.secretary import _round_probs

        assert _round_probs(None) == {}
        assert _round_probs("x") == {}


class TestPerformanceOptimizations:
    """性能优化的行为回归：缓存不能给出过期值，时间格式不能变。"""

    def test_absolute_time_matches_original_formula(self):
        """分钟级缓存后的输出必须与原始实现逐字一致。"""
        from datetime import datetime

        from astrbot_plugin_angel_heart.core.utils.time_utils import (
            format_absolute_time,
        )

        for ts in (1700000000.0, 1700000059.9, 1700001234.5, 1.0, 999999999.0):
            expected = (
                f" ({datetime.fromtimestamp(ts).astimezone().strftime('%Y-%m-%d %H:%M')})"
            )
            assert format_absolute_time(ts) == expected, ts

    def test_absolute_time_invalid_inputs(self):
        from astrbot_plugin_angel_heart.core.utils.time_utils import (
            format_absolute_time,
        )

        assert format_absolute_time(0) == ""
        assert format_absolute_time(-1) == ""
        assert format_absolute_time("bad") == ""

    def test_resolve_override_reflects_template_update(self):
        """模板改了之后，缓存必须失效。"""
        from astrbot_plugin_angel_heart.core.chat_profile import ChatProfileStore

        store = ChatProfileStore(tempfile.mkdtemp())
        tpl = store.create_template("t", config={"timing": {"waiting_time": 3.0}})
        store.set_binding("chat:g:1", tpl["id"])
        assert store.resolve_override("chat:g:1")["timing"]["waiting_time"] == 3.0

        store.update_template(
            tpl["id"], {"config": {"timing": {"waiting_time": 9.0}}}
        )
        assert store.resolve_override("chat:g:1")["timing"]["waiting_time"] == 9.0

    def test_resolve_override_reflects_binding_change(self):
        from astrbot_plugin_angel_heart.core.chat_profile import ChatProfileStore

        store = ChatProfileStore(tempfile.mkdtemp())
        a = store.create_template("a", config={"timing": {"waiting_time": 1.0}})
        b = store.create_template("b", config={"timing": {"waiting_time": 2.0}})
        store.set_binding("chat:g:1", a["id"])
        assert store.resolve_override("chat:g:1")["timing"]["waiting_time"] == 1.0

        store.set_binding("chat:g:1", b["id"])
        assert store.resolve_override("chat:g:1")["timing"]["waiting_time"] == 2.0

        store.set_binding("chat:g:1", "")
        assert store.resolve_override("chat:g:1") is None

    def test_resolve_override_reflects_template_delete(self):
        from astrbot_plugin_angel_heart.core.chat_profile import ChatProfileStore

        store = ChatProfileStore(tempfile.mkdtemp())
        tpl = store.create_template("t", config={"timing": {"waiting_time": 3.0}})
        store.set_binding("chat:g:1", tpl["id"])
        assert store.resolve_override("chat:g:1") is not None

        store.delete_template(tpl["id"])
        assert store.resolve_override("chat:g:1") is None

    def test_config_view_reads_template_override(self):
        """视图内缓存的模板覆盖值仍然正确。"""
        from astrbot_plugin_angel_heart.core.chat_profile import ChatProfileStore
        from astrbot_plugin_angel_heart.core.config_manager import ConfigManager

        store = ChatProfileStore(tempfile.mkdtemp())
        tpl = store.create_template(
            "t", config={"wake_interaction": {"alias": "小天使"}}
        )
        store.set_binding("chat:g:1", tpl["id"])
        cm = ConfigManager({"wake_interaction": {"alias": "默认"}})
        cm.attach_profile_store(store)

        view = cm.for_chat("chat:g:1")
        assert view.alias == "小天使"
        assert view.alias == "小天使"  # 第二次读走缓存，结果一致
        assert cm.for_chat("chat:g:2").alias == "默认"

    def test_transport_request_body_is_plain_json(self):
        """请求体改成预序列化字节后，线上内容必须与 json= 完全一致。"""
        import json as _json

        from astrbot_plugin_angel_heart.core import jev_transport as T

        payload = {"model": "jev-latest", "state": "状态", "questions": {"q": {"type": "noul"}}}
        expected = _json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode()
        assert T.json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode() == expected


class TestCustomPrompt:
    """Jev 提示词可以自己改。"""

    def test_default_prompt_used_when_empty(self):
        from astrbot_plugin_angel_heart.core.jev_decider import (
            DEFAULT_SHOULD_REPLY_INSTRUCTIONS,
            build_questions,
        )

        questions = build_questions("")
        assert questions["should_reply"]["instructions"] == DEFAULT_SHOULD_REPLY_INSTRUCTIONS
        assert build_questions()["should_reply"]["instructions"] == DEFAULT_SHOULD_REPLY_INSTRUCTIONS

    def test_custom_prompt_replaces_main_question(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        questions = build_questions("  只在有人点名时才回话  ")
        assert questions["should_reply"]["instructions"] == "只在有人点名时才回话"
        # 其余判断不受影响
        assert questions["is_questioned"]["instructions"]
        assert questions["reply_strategy"]["type"] == "choice"

    def test_custom_prompt_keeps_question_shape(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_questions

        question = build_questions("自定义")["should_reply"]
        assert question["type"] == "noul"
        assert "true" in question["criteria"]
