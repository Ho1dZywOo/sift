"""OpenCode-backed LLM transport: schema prompts, strict validation, no CC fallback.

与 ``test_dsh_llm_transport.py`` 镜像:mock ``opencode_llm.complete``,锁三条主线——
① resolve 支持 opencode(+default 参数的决策链语义);② schema 模式的 prompt 注入/
fence 剥离/jsonschema 校验/items 解包;③ 文本模式透传与 usage 回调形状。全部
不真跑 opencode。
"""

from __future__ import annotations

import json

import pytest

from sift import llm_transport as transport
from sift.execution import opencode_llm
from sift.llm_transport import LLMCallConfig


def _completion(text: str, *, inp: int = 0, out: int = 0, cost: float = 0.0, reason: str = "stop"):
    return opencode_llm.OpencodeCompletion(
        text=text, input_tokens=inp, output_tokens=out, cost_usd=cost, stop_reason=reason
    )


def test_resolve_llm_transport_supports_opencode(monkeypatch):
    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "opencode")
    assert transport.resolve_llm_transport(None) == "opencode"
    assert transport.resolve_llm_transport("opencode") == "opencode"


def test_resolve_llm_transport_default_param_and_env_precedence(monkeypatch):
    """site default 经 default 参数表达;env 覆盖 site default;显式值覆盖 env。"""
    monkeypatch.delenv("SIFT_LLM_TRANSPORT", raising=False)
    assert transport.resolve_llm_transport(None, default="http") == "http"
    monkeypatch.setenv("SIFT_LLM_TRANSPORT", "opencode")
    assert transport.resolve_llm_transport(None, default="http") == "opencode"  # env 胜 site default
    assert transport.resolve_llm_transport("cc", default="http") == "cc"  # 显式胜 env
    with pytest.raises(RuntimeError, match="cc / http / dsh / opencode"):
        transport.resolve_llm_transport("carrier-pigeon")


def test_call_claude_opencode_transport_schema_flow(monkeypatch):
    seen: dict[str, object] = {}

    def fake_complete(prompt, *, model=None, timeout=120, on_progress=None):
        seen["prompt"] = prompt
        seen["model"] = model
        seen["timeout"] = timeout
        return _completion('```json\n{"ok": true}\n```', inp=17, out=5, cost=0.02)

    monkeypatch.setattr(opencode_llm, "complete", fake_complete)
    monkeypatch.setattr(transport, "call_cli", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call cc")))
    usage: list[dict] = []

    result = transport.call_claude(
        prompt="answer this",
        config=LLMCallConfig(model="gateway/glm-5.3", timeout=33, schema={"type": "object"}, transport="opencode"),
        on_usage=usage.append,
    )

    assert json.loads(result) == {"ok": True}
    assert "JSON Schema" in str(seen["prompt"])  # schema 内嵌进 prompt(dsh 同型)
    assert "Do not call tools" in str(seen["prompt"])
    assert seen["model"] == "gateway/glm-5.3"
    assert seen["timeout"] == 33
    # usage payload 形状对齐 call_claude 契约(input/output/cost)
    assert usage == [{"input_tokens": 17, "output_tokens": 5, "total_cost_usd": 0.02}]


def test_call_claude_opencode_transport_text_mode_passthrough(monkeypatch):
    monkeypatch.setattr(opencode_llm, "complete", lambda prompt, **k: _completion("plain answer"))
    out = transport.call_claude(
        prompt="say hi", config=LLMCallConfig(transport="opencode"), on_usage=lambda u: None
    )
    assert out == "plain answer"  # schema=None → 纯文本进出,不做校验


def test_call_claude_opencode_transport_rejects_invalid_json(monkeypatch):
    monkeypatch.setattr(opencode_llm, "complete", lambda prompt, **k: _completion("not json"))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        transport.call_claude(
            prompt="x",
            config=LLMCallConfig(schema={"type": "object"}, transport="opencode"),
        )


def test_call_claude_opencode_transport_rejects_schema_invalid(monkeypatch):
    monkeypatch.setattr(opencode_llm, "complete", lambda prompt, **k: _completion('{"wrong": 1}'))
    with pytest.raises(RuntimeError, match="schema-invalid"):
        transport.call_claude(
            prompt="x",
            config=LLMCallConfig(schema={"type": "object", "required": ["verdict"]}, transport="opencode"),
        )


def test_call_claude_opencode_transport_unwraps_items_array(monkeypatch):
    """array 判官约定:{"items":[...]} → 裸 array(与 call_cli/call_http 同约)。"""
    monkeypatch.setattr(
        opencode_llm, "complete", lambda prompt, **k: _completion('{"items": [{"verdict": "PASS"}, {"verdict": "FAIL"}]}')
    )
    result = transport.call_claude(
        prompt="x",
        config=LLMCallConfig(
            schema={"type": "object", "properties": {"items": {"type": "array"}}}, transport="opencode"
        ),
    )
    assert json.loads(result) == [{"verdict": "PASS"}, {"verdict": "FAIL"}]


def test_call_claude_opencode_transport_empty_text_raises(monkeypatch):
    monkeypatch.setattr(opencode_llm, "complete", lambda prompt, **k: _completion("   "))
    with pytest.raises(RuntimeError, match="empty final text"):
        transport.call_claude(prompt="x", config=LLMCallConfig(transport="opencode"))


def test_call_dsh_items_array_unwrap_parity(monkeypatch):
    """dsh 与 opencode 的 items 解包对齐(call_cli/call_http 原有约定,回归锁)。"""
    from sift.execution.dsh_runner import DshRunner
    from sift.models import RunOutput

    monkeypatch.setattr(
        DshRunner,
        "complete",
        staticmethod(
            lambda prompt, options: RunOutput(
                exit_code=0, stdout="", total_tokens=0, result_text='{"items": [1, 2]}', stop_reason="ok"
            )
        ),
    )
    result = transport.call_claude(
        prompt="x",
        config=LLMCallConfig(
            schema={"type": "object", "properties": {"items": {"type": "array"}}}, transport="dsh"
        ),
    )
    assert json.loads(result) == [1, 2]
