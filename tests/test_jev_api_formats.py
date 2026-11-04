"""接口协议切换：SystemOne ↔ OpenAI Decisions 的翻译。

内部统一用 SystemOne 形状（判定层不改），只在发包/收包时翻译。
"""

import json

import pytest

from astrbot_plugin_angel_heart.core.jev_transport import (
    API_FORMAT_DECISIONS,
    API_FORMAT_SYSTEMONE,
    build_payload,
    normalize_base_url,
    resolve_api_format,
    to_systemone_response,
    valid_answers,
)

QUESTIONS = {
    "should_reply": {"type": "noul", "instructions": "要不要接话？"},
    "route": {
        "type": "choice",
        "instructions": "去哪个队列？",
        "criteria": {"billing": "账单", "tech": "技术"},
    },
    "urgency": {
        "type": "score",
        "instructions": "有多急？",
        "criteria": ["低", "中", "高"],
    },
}


class TestResolveFormat:
    def test_default_is_systemone(self):
        assert resolve_api_format({}) == API_FORMAT_SYSTEMONE
        assert resolve_api_format(None) == API_FORMAT_SYSTEMONE

    def test_explicit_decisions(self):
        assert resolve_api_format({"api_format": "decisions"}) == API_FORMAT_DECISIONS

    def test_aliases(self):
        for alias in ("openai", "openai_decisions", "OpenAI", "DECISIONS"):
            assert resolve_api_format({"api_format": alias}) == API_FORMAT_DECISIONS

    def test_unknown_falls_back(self):
        assert resolve_api_format({"api_format": "???"}) == API_FORMAT_SYSTEMONE


class TestNormalizeBaseUrl:
    def test_systemone_default(self):
        assert (
            normalize_base_url("https://jevmodel.org")
            == "https://jevmodel.org/v1/systemone"
        )

    def test_decisions(self):
        assert (
            normalize_base_url("https://api.example.com", API_FORMAT_DECISIONS)
            == "https://api.example.com/v1/decisions"
        )

    def test_already_full_path(self):
        assert (
            normalize_base_url("https://jevmodel.org/v1/systemone")
            == "https://jevmodel.org/v1/systemone"
        )

    def test_switch_format_replaces_old_suffix(self):
        """地址里已经带了 systemone 尾缀，切协议时要换成 decisions。"""
        assert (
            normalize_base_url("https://api.example.com/v1/systemone", API_FORMAT_DECISIONS)
            == "https://api.example.com/v1/decisions"
        )

    def test_trailing_v1(self):
        assert (
            normalize_base_url("https://api.example.com/v1", API_FORMAT_DECISIONS)
            == "https://api.example.com/v1/decisions"
        )

    def test_trailing_slash(self):
        assert (
            normalize_base_url("https://api.example.com/", API_FORMAT_SYSTEMONE)
            == "https://api.example.com/v1/systemone"
        )


class TestBuildPayload:
    def test_systemone_unchanged(self):
        payload = build_payload("m", "state", QUESTIONS, API_FORMAT_SYSTEMONE)
        assert payload == {"model": "m", "state": "state", "questions": QUESTIONS}

    def test_decisions_shape(self):
        payload = build_payload("m", "hello", QUESTIONS, API_FORMAT_DECISIONS)
        assert payload["model"] == "m"
        assert payload["input"] == "hello"
        assert "state" not in payload
        # questions 是数组，按请求顺序
        assert [q["name"] for q in payload["questions"]] == list(QUESTIONS)

    def test_decisions_noul_becomes_predicate(self):
        payload = build_payload("m", "x", QUESTIONS, API_FORMAT_DECISIONS)
        by_name = {q["name"]: q for q in payload["questions"]}
        assert by_name["should_reply"]["type"] == "predicate"

    def test_decisions_criteria_becomes_choices(self):
        payload = build_payload("m", "x", QUESTIONS, API_FORMAT_DECISIONS)
        by_name = {q["name"]: q for q in payload["questions"]}
        assert by_name["route"]["choices"] == [
            {"value": "billing", "description": "账单"},
            {"value": "tech", "description": "技术"},
        ]

    def test_decisions_score_criteria_becomes_levels(self):
        payload = build_payload("m", "x", QUESTIONS, API_FORMAT_DECISIONS)
        by_name = {q["name"]: q for q in payload["questions"]}
        assert by_name["urgency"]["levels"] == [
            {"label": "低"}, {"label": "中"}, {"label": "高"}
        ]

    def test_structured_state_is_serialized(self):
        payload = build_payload("m", {"文本": "你好"}, QUESTIONS, API_FORMAT_DECISIONS)
        assert isinstance(payload["input"], str)
        assert "你好" in payload["input"]


class TestToSystemoneResponse:
    def test_choice(self):
        data = {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "name": "route",
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.93,
                    "probabilities": [
                        {"value": "billing", "probability": 0.93},
                        {"value": "tech", "probability": 0.07},
                    ],
                }
            ],
        }
        out = to_systemone_response(data)
        answer = out["answers"]["route"]
        assert answer["type"] == "choice"
        assert answer["choice"] == "billing"
        assert answer["probabilities"] == {"billing": 0.93, "tech": 0.07}
        assert answer["confidence"] == 0.93

    def test_predicate_becomes_noul(self):
        data = {
            "answers": [
                {"name": "should_reply", "type": "predicate", "probability": 0.12}
            ]
        }
        out = to_systemone_response(data)
        assert out["answers"]["should_reply"] == {"type": "noul", "noul": 0.12}

    def test_score(self):
        data = {
            "answers": [
                {"name": "urgency", "type": "score", "score": 1.4, "confidence": 0.88}
            ]
        }
        out = to_systemone_response(data)
        assert out["answers"]["urgency"]["type"] == "score"
        assert out["answers"]["urgency"]["score"] == 1.4

    def test_refusal_is_dropped(self):
        data = {
            "answers": [
                {"name": "route", "type": "refusal"},
                {
                    "name": "should_reply",
                    "type": "predicate",
                    "probability": 0.9,
                },
            ]
        }
        out = to_systemone_response(data)
        assert "route" not in out["answers"]
        assert out["answers"]["should_reply"]["noul"] == 0.9

    def test_systemone_response_untouched(self):
        data = {"answers": {"route": {"type": "choice", "choice": "billing"}}}
        assert to_systemone_response(data) == data


class TestValidAnswers:
    def test_require_all_default(self):
        data = {"answers": {"should_reply": {"type": "noul", "noul": 0.5}}}
        assert valid_answers(data, QUESTIONS) is False

    def test_partial_allowed(self):
        """Decisions 允许单题拒答，翻译后那题会被跳过——不能因此判整份无效。"""
        data = {"answers": {"should_reply": {"type": "noul", "noul": 0.5}}}
        assert valid_answers(data, QUESTIONS, require_all=False) is True

    def test_wrong_type_rejected(self):
        data = {"answers": {"should_reply": {"type": "choice", "choice": "x"}}}
        assert valid_answers(data, QUESTIONS, require_all=False) is False


class TestRoundTrip:
    def test_decisions_request_then_response(self):
        """发出去的包是 Decisions 形状，收回来还是 SystemOne 形状。"""
        payload = build_payload("m", "hi", QUESTIONS, API_FORMAT_DECISIONS)
        # 发出去的包不能带 SystemOne 专属字段
        assert "state" not in payload
        assert isinstance(payload["questions"], list)

        # 假装网关回了 Decisions 的答案
        upstream = {
            "model": "m",
            "answers": [
                {"name": "should_reply", "type": "predicate", "probability": 0.8},
                {
                    "name": "route",
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "probabilities": [{"value": "billing", "probability": 0.9}],
                },
                {"name": "urgency", "type": "score", "score": 2.0, "confidence": 0.7},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }
        translated = to_systemone_response(upstream)
        assert valid_answers(translated, QUESTIONS, require_all=False) is True
        # 上层拿到的还是「以问题名为键」的映射
        assert set(translated["answers"]) == set(QUESTIONS)
        assert translated["answers"]["should_reply"]["noul"] == 0.8
        assert json.dumps(translated, ensure_ascii=False)


class TestCustomEndpointPath:
    """第三方代理（如 jevtypesafeai.com）用的是自定义路径。"""

    def test_custom_path(self):
        assert (
            normalize_base_url(
                "https://jevtypesafeai.com", API_FORMAT_SYSTEMONE, "/api/v1/decide"
            )
            == "https://jevtypesafeai.com/api/v1/decide"
        )

    def test_custom_path_without_leading_slash(self):
        assert (
            normalize_base_url(
                "https://jevtypesafeai.com", API_FORMAT_SYSTEMONE, "api/v1/decide"
            )
            == "https://jevtypesafeai.com/api/v1/decide"
        )

    def test_custom_path_wins_over_protocol(self):
        """自定义路径优先于协议默认路径。"""
        assert normalize_base_url(
            "https://x.dev", API_FORMAT_DECISIONS, "/api/v1/decide"
        ).endswith("/api/v1/decide")

    def test_full_url_in_base_is_kept(self):
        assert (
            normalize_base_url(
                "https://jevtypesafeai.com/api/v1/decide",
                API_FORMAT_SYSTEMONE,
                "/api/v1/decide",
            )
            == "https://jevtypesafeai.com/api/v1/decide"
        )

    def test_empty_path_uses_protocol(self):
        assert normalize_base_url("https://x.dev", API_FORMAT_DECISIONS, "").endswith(
            "/v1/decisions"
        )

    def test_resolve_endpoint_path(self):
        from astrbot_plugin_angel_heart.core.jev_transport import resolve_endpoint_path

        assert resolve_endpoint_path({"endpoint_path": " /api/v1/decide "}) == "/api/v1/decide"
        assert resolve_endpoint_path({}) == ""
        assert resolve_endpoint_path(None) == ""

    def test_gateway_preset(self):
        from astrbot_plugin_angel_heart.core.jev_transport import GATEWAYS

        assert GATEWAYS["jevtypesafeai"] == "https://jevtypesafeai.com"


class TestNewErrorCodes:
    def test_403_and_413(self):
        from astrbot_plugin_angel_heart.core.jev_transport import error_status

        assert error_status(403) == "forbidden"
        assert error_status(413) == "payload_too_large"
