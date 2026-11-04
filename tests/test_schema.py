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
