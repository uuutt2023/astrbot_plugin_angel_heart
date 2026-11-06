"""配置页里 Jev 那组的关键项不能走样。"""

import json
from pathlib import Path

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
)
JEV = SCHEMA["jev_decision"]["items"]

EXPECTED_PROTOCOLS = ["systemone", "decisions", "custom"]


def test_proxy_protocol_is_a_fixed_dropdown():
    item = JEV["proxy_protocol"]
    assert item["type"] == "string"
    assert item["options"] == EXPECTED_PROTOCOLS
    assert len(item["labels"]) == len(EXPECTED_PROTOCOLS)
    assert item["default"] == "systemone"


def test_client_only_keeps_gateway_address_and_protocol():
    """Key、地址、协议、超时这些都归网关，插件这边不该再有。"""
    for key in ("api_key", "base_url", "api_format", "endpoint_path", "model", "timeout_sec", "retries"):
        assert key not in JEV, f"客户端不该再有 {key}"


def test_gateway_address_is_still_configurable():
    assert JEV["proxy_url"]["type"] == "string"


def test_jev_prompt_is_customizable():
    """Jev 提示词要能自己改，类型是 text（渲染成大文本框）。"""
    item = JEV["prompt"]
    assert item["type"] == "text"
    assert item["default"] == ""
    assert item["editor_mode"] is True


def test_image_handling_switches_exist():
    """图片处理两个开关都在，且默认关闭（保持原行为）。"""
    ignore = JEV["ignore_image_judgment"]
    transcribe = JEV["transcribe_image_with_model"]
    assert ignore["type"] == "bool"
    assert ignore["default"] is False
    assert transcribe["type"] == "bool"
    assert transcribe["default"] is False
    assert ignore["hint"] and transcribe["hint"]
