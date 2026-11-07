"""钉死 Jev 判定前的图片处理：模式解析、消息预处理与秘书分支。

覆盖：
- resolve_image_mode：默认原样；忽视优先于转述；
- message_has_image / message_text / is_image_only：图片识别与占位剥离；
- prepare_messages_for_jev：忽视模式清占位、转述模式附上转述、原样模式不动原消息；
- Secretary：忽视模式下纯图片消息不调 Jev 且直接不回复；转述模式走图片理解模型。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

PLUGIN_ROOT = None


def make_image_msg(name: str = "A", text: str = "[图片]", ts: float = 0.0) -> dict:
    """构造一条带图片的消息。"""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.jpg"},
                "cache_path": "/tmp/angel/a.jpg",
            },
        ],
        "sender_name": name,
        "sender_id": name,
        "timestamp": ts,
    }


def make_text_msg(name: str = "A", text: str = "在吗", ts: float = 0.0) -> dict:
    """构造一条纯文字消息。"""
    return {
        "role": "user",
        "content": text,
        "sender_name": name,
        "sender_id": name,
        "timestamp": ts,
    }


class TestResolveImageMode:
    """模式解析。"""

    def test_default_is_keep(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_KEEP, resolve_image_mode

        assert resolve_image_mode({}) == MODE_KEEP
        assert resolve_image_mode(None) == MODE_KEEP

    def test_ignore_flag(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_IGNORE, resolve_image_mode

        assert resolve_image_mode({"ignore_image_judgment": True}) == MODE_IGNORE

    def test_transcribe_flag(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_CAPTION, resolve_image_mode

        cfg = {"transcribe_image_with_model": True}
        assert resolve_image_mode(cfg) == MODE_CAPTION

    def test_ignore_wins_over_transcribe(self):
        """忽视优先：两个开关都开时走忽视。"""
        from astrbot_plugin_angel_heart.core.jev_images import MODE_IGNORE, resolve_image_mode

        cfg = {"ignore_image_judgment": True, "transcribe_image_with_model": True}
        assert resolve_image_mode(cfg) == MODE_IGNORE


class TestMessageIntrospection:
    """图片识别与正文提取。"""

    def test_has_image(self):
        from astrbot_plugin_angel_heart.core.jev_images import message_has_image

        assert message_has_image(make_image_msg()) is True
        assert message_has_image(make_text_msg()) is False

    def test_text_strips_placeholder(self):
        from astrbot_plugin_angel_heart.core.jev_images import message_text

        assert message_text(make_image_msg(text="[图片]")) == ""
        assert message_text(make_image_msg(text="[图片] 哈哈哈")) == "哈哈哈"
        assert message_text(make_text_msg(text="在吗")) == "在吗"

    def test_image_only_detection(self):
        from astrbot_plugin_angel_heart.core.jev_images import is_image_only

        assert is_image_only(make_image_msg(text="[图片]")) is True
        assert is_image_only(make_image_msg(text="[图片] 看看这个")) is False
        assert is_image_only(make_text_msg()) is False


class TestPrepareMessages:
    """消息预处理。"""

    @pytest.mark.asyncio
    async def test_keep_mode_keeps_everything(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_KEEP, prepare_messages_for_jev

        msg = make_image_msg()
        out = await prepare_messages_for_jev([msg], MODE_KEEP)
        assert out[0] is msg

    @pytest.mark.asyncio
    async def test_ignore_mode_strips_placeholder_and_keeps_text(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_IGNORE, prepare_messages_for_jev

        msg = make_image_msg(text="[图片] 魔女帽里没魔女")
        out = await prepare_messages_for_jev([msg], MODE_IGNORE)
        assert out[0]["content"] == "魔女帽里没魔女"
        # 原消息不能被改
        assert msg["content"][0]["text"] == "[图片] 魔女帽里没魔女"

    @pytest.mark.asyncio
    async def test_ignore_mode_drops_image_component(self):
        """图片组件不该进 Jev 上下文（它没有文字价值）。"""
        from astrbot_plugin_angel_heart.core.jev_images import MODE_IGNORE, prepare_messages_for_jev

        out = await prepare_messages_for_jev([make_image_msg()], MODE_IGNORE)
        assert out[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_caption_mode_appends_caption(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_CAPTION, prepare_messages_for_jev

        async def captioner(msg):
            return "一只橘猫趴在键盘上"

        out = await prepare_messages_for_jev([make_image_msg(text="[图片]")], MODE_CAPTION, captioner)
        assert out[0]["content"] == "[图片: 一只橘猫趴在键盘上]"

    @pytest.mark.asyncio
    async def test_caption_mode_marks_unreadable(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_CAPTION, prepare_messages_for_jev

        async def captioner(msg):
            return ""

        out = await prepare_messages_for_jev([make_image_msg(text="[图片]")], MODE_CAPTION, captioner)
        assert out[0]["content"] == "[图片无法读取]"

    @pytest.mark.asyncio
    async def test_captioner_failure_does_not_break(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_CAPTION, prepare_messages_for_jev

        async def captioner(msg):
            raise RuntimeError("视觉模型挂了")

        out = await prepare_messages_for_jev([make_image_msg(text="[图片]")], MODE_CAPTION, captioner)
        assert out[0]["content"] == "[图片无法读取]"

    @pytest.mark.asyncio
    async def test_text_messages_untouched(self):
        from astrbot_plugin_angel_heart.core.jev_images import MODE_IGNORE, prepare_messages_for_jev

        msg = make_text_msg(text="在吗")
        out = await prepare_messages_for_jev([msg], MODE_IGNORE)
        assert out[0] is msg


def make_config(**jev_overrides):
    """构造带 Jev 配置的 ConfigManager。"""
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


class TestSecretaryImageHandling:
    """秘书按图片模式决定要不要问 Jev。"""

    def _secretary(self, config, ledger=None):
        from astrbot_plugin_angel_heart.roles.secretary import Secretary

        angel_context = MagicMock()
        angel_context.conversation_ledger = ledger
        return Secretary(config, context=MagicMock(), angel_context=angel_context)

    @pytest.mark.asyncio
    async def test_ignore_mode_skips_jev_for_image_only(self):
        """忽视图片：纯图片消息不调 Jev，直接不回复。"""
        secretary = self._secretary(make_config(ignore_image_judgment=True))
        secretary.jev_decider.decide = AsyncMock()

        decision = await secretary.perform_analysis([make_image_msg()], [], "chat")
        assert decision.should_reply is False
        assert secretary.jev_decider.decide.await_count == 0

    @pytest.mark.asyncio
    async def test_keep_mode_still_asks_jev_for_image_only(self):
        """默认模式保持原行为：图片消息照常问 Jev。"""
        secretary = self._secretary(make_config())
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

        decision = await secretary.perform_analysis([make_image_msg()], [], "chat")
        assert secretary.jev_decider.decide.await_count == 1
        assert decision.should_reply is True

    @pytest.mark.asyncio
    async def test_ignore_mode_keeps_text_message(self):
        """带文字的图片消息：只清图片占位，文字照常问 Jev。"""
        secretary = self._secretary(make_config(ignore_image_judgment=True))
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

        await secretary.perform_analysis([make_image_msg(text="[图片] 魔女帽里没魔女")], [], "chat")
        assert secretary.jev_decider.decide.await_count == 1
        sent = secretary.jev_decider.decide.await_args.args[1]
        assert sent[0]["content"] == "魔女帽里没魔女"

    @pytest.mark.asyncio
    async def test_ignore_mode_hybrid_gate_blocks(self):
        """hybrid 门控：纯图片消息直接不回复，不放行给秘书模型。"""
        secretary = self._secretary(
            make_config(backend="hybrid", ignore_image_judgment=True)
        )
        secretary.llm_analyzer.analyze_and_decide = AsyncMock()
        secretary.jev_decider.decide = AsyncMock()

        decision = await secretary.perform_analysis([make_image_msg()], [], "chat")
        assert decision.should_reply is False
        assert secretary.jev_decider.decide.await_count == 0
        assert secretary.llm_analyzer.analyze_and_decide.await_count == 0

    @pytest.mark.asyncio
    async def test_caption_mode_uses_vision_model(self):
        """转述模式：调插件的图片理解模型，把转述写进 Jev 上下文。"""
        ledger = MagicMock()
        ledger.caption_message_images = AsyncMock(return_value="一只橘猫趴在键盘上")
        secretary = self._secretary(make_config(transcribe_image_with_model=True), ledger)
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

        await secretary.perform_analysis([make_image_msg()], [], "chat")
        assert ledger.caption_message_images.await_count == 1
        sent = secretary.jev_decider.decide.await_args.args[1]
        assert sent[0]["content"] == "[图片: 一只橘猫趴在键盘上]"

    @pytest.mark.asyncio
    async def test_caption_mode_without_provider_still_asks_jev(self):
        """没配图片理解模型时降级为「图片无法读取」，不阻断判定。"""
        ledger = MagicMock()
        ledger.caption_message_images = AsyncMock(return_value="")
        secretary = self._secretary(make_config(transcribe_image_with_model=True), ledger)
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

        await secretary.perform_analysis([make_image_msg()], [], "chat")
        assert secretary.jev_decider.decide.await_count == 1
        sent = secretary.jev_decider.decide.await_args.args[1]
        assert sent[0]["content"] == "[图片无法读取]"


LEDGER_SAMPLE = """助理工作账本：
- [本轮] 触发锚点=-1144523218；任务=AI时代的Steam管家出现了 [图片] [图片]
- [已完成] 触发锚点=63702767；任务=这叫暗访记者；结果=不回复：继续观察
- [已完成] 触发锚点=-2138075607；任务=[图片]；结果=不回复：图片消息（已忽视）
- [已完成] 触发锚点=605605276；任务=[At:327778013] 打麻将；结果=不回复：继续观察
- [已完成] 触发锚点=1526790835；任务=[图片]；结果=不回复：图片消息（已忽视）
本轮待处理：AI时代的Steam管家出现了 [图片] [图片]。这是当前事件对应的工作，可以继续决策。"""


class TestStripLedgerImages:
    """账本里的图片占位要一并剔除。"""

    def test_placeholders_removed(self):
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        out = strip_ledger_images(LEDGER_SAMPLE)
        assert "[图片]" not in out
        assert "任务=AI时代的Steam管家出现了" in out

    def test_image_only_entries_dropped(self):
        """任务只剩图片的条目整条去掉。"""
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        out = strip_ledger_images(LEDGER_SAMPLE)
        assert "图片消息（已忽视）" not in out
        assert "触发锚点=-2138075607" not in out
        assert "触发锚点=1526790835" not in out

    def test_other_entries_kept(self):
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        out = strip_ledger_images(LEDGER_SAMPLE)
        assert "这叫暗访记者" in out
        assert "打麻将" in out
        assert "触发锚点=63702767" in out

    def test_pending_line_kept_without_images(self):
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        out = strip_ledger_images(LEDGER_SAMPLE)
        assert "本轮待处理：AI时代的Steam管家出现了。这是当前事件对应的工作，可以继续决策。" in out

    def test_all_entries_gone_returns_empty(self):
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        text = "助理工作账本：\n- [已完成] 触发锚点=1；任务=[图片]；结果=不回复"
        assert strip_ledger_images(text) == ""

    def test_empty_input(self):
        from astrbot_plugin_angel_heart.core.jev_images import strip_ledger_images

        assert strip_ledger_images("") == ""
        assert strip_ledger_images("   ") == ""


class TestDecideStripsLedgerImages:
    """忽视图片时，账本里的图片信息不进上下文。"""

    def _config(self, **jev_overrides):
        from astrbot_plugin_angel_heart.core.config_manager import ConfigManager

        jev = {
            "enabled": True,
            "proxy_url": "http://127.0.0.1:8770",
            "proxy_protocol": "systemone",
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

    async def _state_with_ledger(self, cfg):
        from astrbot_plugin_angel_heart.core.jev_decider import JevDecider

        decider = JevDecider()
        captured = {}

        async def fake_evaluate(cfg, state, questions, idem=None):
            captured["state"] = state
            return {
                "status": "ready",
                "answers": {"should_reply": {"type": "noul", "noul": 0.9}},
            }

        decider.client.evaluate = fake_evaluate
        try:
            await decider.decide(
                cfg,
                [make_text_msg(text="AI时代的Steam管家出现了")],
                work_ledger_text=LEDGER_SAMPLE,
            )
        finally:
            await decider.close()
        return captured["state"]

    @pytest.mark.asyncio
    async def test_ignore_mode_strips_ledger_images(self):
        state = await self._state_with_ledger(
            self._config(ignore_image_judgment=True).jev_decision
        )
        assert "[图片]" not in state
        assert "图片消息（已忽视）" not in state
        assert "本轮待处理：AI时代的Steam管家出现了。" in state

    @pytest.mark.asyncio
    async def test_keep_mode_keeps_ledger_images(self):
        """默认模式不动账本。"""
        state = await self._state_with_ledger(self._config().jev_decision)
        assert "[图片]" in state
