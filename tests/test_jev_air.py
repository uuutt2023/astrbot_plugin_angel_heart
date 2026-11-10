"""钉死 Jev 的读空气增强：上下文信号、state 接入与开关。

覆盖：
- jev_air：当前发送者、@目标、连续回复轮次、拒绝信号、助理自己的历史回复；
- build_state：读空气块进上下文、裁剪时保留、助理回复带标记；
- JevDecider.decide：读空气开关开时带上信号，关时行为与以前一致。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def user_msg(name="A", text="在吗", sid="111", ts=1.0, **extra):
    """构造一条用户消息。"""
    msg = {
        "role": "user",
        "content": text,
        "sender_name": name,
        "sender_id": sid,
        "timestamp": ts,
    }
    msg.update(extra)
    return msg


def assistant_msg(text="我来帮你看看", ts=2.0):
    """构造一条助理自己的消息。"""
    return {
        "role": "assistant",
        "content": text,
        "sender_name": "assistant",
        "sender_id": "0",
        "timestamp": ts,
    }


class TestAirSignals:
    """读空气信号的提取。"""

    def test_current_sender_skips_assistant(self):
        from astrbot_plugin_angel_heart.core.jev_air import current_sender

        msg = user_msg(name="小王", sid="888")
        assert current_sender([assistant_msg(), msg])["sender_name"] == "小王"

    def test_at_targets(self):
        from astrbot_plugin_angel_heart.core.jev_air import at_targets

        assert at_targets(user_msg(text="@李四 你看这个")) == ["李四"]
        assert at_targets(user_msg(text="没有提及")) == []

    def test_trailing_assistant_rounds(self):
        from astrbot_plugin_angel_heart.core.jev_air import trailing_assistant_rounds

        assert trailing_assistant_rounds([user_msg(), assistant_msg()]) == 1
        assert trailing_assistant_rounds([assistant_msg(), assistant_msg()]) == 2
        assert trailing_assistant_rounds([assistant_msg(), user_msg()]) == 0

    def test_refusal_signals(self):
        from astrbot_plugin_angel_heart.core.jev_air import refusal_signals

        hits = refusal_signals([user_msg(text="别烦我")], [])
        assert "别烦我" in hits

    def test_assistant_replies_only_own(self):
        from astrbot_plugin_angel_heart.core.jev_air import assistant_replies

        replies = assistant_replies([user_msg(), assistant_msg("之前说过的话")], [])
        assert replies == ["之前说过的话"]


class TestBuildAirContext:
    """读空气上下文块。"""

    def test_empty_without_messages(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        assert build_air_context([], []) == ""

    def test_contains_sender_and_audience(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        air = build_air_context([user_msg(name="小王", sid="888", text="@李四 你看")], [])
        assert "[Air reading]" in air
        assert "小王" in air and "888" in air
        assert "@李四" in air

    def test_at_self_marked(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        air = build_air_context([user_msg(text="@机器人 在吗", is_at_self=True)], [])
        assert "mentions the assistant directly" in air

    def test_fatigue_note(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        air = build_air_context(
            [user_msg(), assistant_msg(), assistant_msg(), assistant_msg()], []
        )
        assert "fatigue" in air

    def test_refusal_in_context(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        air = build_air_context([user_msg(text="别烦我了")], [])
        assert "less talking" in air

    def test_own_replies_listed(self):
        from astrbot_plugin_angel_heart.core.jev_air import build_air_context

        air = build_air_context([user_msg(), assistant_msg("我刚说过这句")], [])
        assert "repeating them adds nothing" in air
        assert "我刚说过这句" in air


class TestStateIntegration:
    """读空气块进 state。"""

    def test_air_context_appended(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state(
            [user_msg(text="在吗")],
            [],
            air_context="[Air reading]\nCurrent sender: 小王 (ID: 888).",
        )
        assert "[Air reading]" in state
        assert state.index("[Current]") < state.index("[Air reading]")

    def test_air_context_survives_truncation(self):
        """裁剪时读空气块不能被裁掉。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        history = [user_msg(name=f"U{i}", text=f"历史消息{i}" * 20, sid=str(i), ts=i) for i in range(20)]
        state = build_state(
            [user_msg(text="在吗")],
            history,
            max_chars=600,
            air_context="[Air reading]\nCurrent sender: 小王 (ID: 888).",
        )
        assert "[Air reading]" in state
        assert len(state) <= 900

    def test_assistant_replies_marked(self):
        """助理自己的历史回复要带标记，模型才分得出谁说的。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state(
            [user_msg(text="在吗"), assistant_msg("我刚说的话")], [], mark_own_replies=True
        )
        assert "[Assistant's own earlier reply]" in state

    def test_marking_off_by_default(self):
        """开关关掉时保持旧行为：不加标记。"""
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state([user_msg(text="在吗"), assistant_msg("我刚说的话")], [])
        assert "[Assistant's own earlier reply]" not in state

    def test_scene_header_mentions_current_first(self):
        from astrbot_plugin_angel_heart.core.jev_decider import build_state

        state = build_state([user_msg(text="在吗")], [])
        assert "placed first" in state
        assert "oldest first" not in state.split("[Current]")[0]


class TestDecideAirSwitch:
    """开关行为。"""

    def _config(self, **jev_overrides):
        from astrbot_plugin_angel_heart.core.config_manager import ConfigManager

        jev = {
            "enabled": True,
            "proxy_url": "http://127.0.0.1:8770",
            "proxy_protocol": "systemone",
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
                "decision_backend": "jev",
                "jev_decision": jev,
                "wake_interaction": {"alias": "fairy"},
                "timing": {},
                "leave_reply": {},
                "access_control": {},
                "context_compression": {},
                "output_rewrite": {},
            }
        )

    @pytest.mark.asyncio
    async def test_air_reading_on_by_default(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        captured = {}

        async def fake_evaluate(cfg, state, questions, idem=None):
            captured["state"] = state
            return {
                "status": "ready",
                "answers": {
                    "should_reply": {"type": "noul", "noul": 0.9},
                    "is_questioned": {"type": "noul", "noul": 0.8},
                    "is_interesting": {"type": "noul", "noul": 0.3},
                    "reply_strategy": {"type": "choice", "choice": "answer_question"},
                },
            }

        decider.client.evaluate = fake_evaluate
        try:
            await decider.decide(
                self._config().jev_decision,
                [user_msg(name="小王", sid="888", text="@李四 你看这个")],
            )
        finally:
            await decider.close()

        assert "[Air reading]" in captured["state"]
        assert "@李四" in captured["state"]

    @pytest.mark.asyncio
    async def test_air_reading_can_be_disabled(self):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        captured = {}

        async def fake_evaluate(cfg, state, questions, idem=None):
            captured["state"] = state
            return {
                "status": "ready",
                "answers": {
                    "should_reply": {"type": "noul", "noul": 0.9},
                    "is_questioned": {"type": "noul", "noul": 0.8},
                    "is_interesting": {"type": "noul", "noul": 0.3},
                    "reply_strategy": {"type": "choice", "choice": "answer_question"},
                },
            }

        decider.client.evaluate = fake_evaluate
        try:
            await decider.decide(
                self._config(air_reading=False).jev_decision,
                [user_msg(name="小王", sid="888", text="@李四 你看这个")],
            )
        finally:
            await decider.close()

        assert "[Air reading]" not in captured["state"]
        assert "[Assistant's own earlier reply]" not in captured["state"]

    def test_default_instructions_carry_air_criteria(self):
        """内置主判定题要带上读空气判据，但不能只强调不回复。"""
        from astrbot_plugin_angel_heart.core.jev_decider import (
            DEFAULT_SHOULD_REPLY_INSTRUCTIONS,
        )

        lowered = DEFAULT_SHOULD_REPLY_INSTRUCTIONS.lower()
        assert "meant for someone else" in lowered
        assert "has already said the same thing" in lowered
        assert "asked for less talking" in lowered
        # 普通聊天也该算作开口的理由，不能只认「被点名」
        assert "ordinary" in lowered
        assert "a member of this group" in lowered
