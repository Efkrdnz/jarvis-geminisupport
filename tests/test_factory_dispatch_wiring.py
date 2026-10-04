"""Integration tests: each migrated module's local LLM wrapper actually
dispatches through ``get_llm_backend(cfg)``.

Existing unit tests patch the module-local ``call_llm_direct`` /
``chat_with_messages`` symbol — that's the right boundary for behaviour
tests, but it leaves a hole: if a wrapper accidentally drops the
``get_llm_backend(cfg)`` call (or hard-codes ``OllamaBackend``), every
unit test still passes because they never reach the dispatch shim.

These tests close that hole by patching the *backend constructors* and
asserting each wrapper picks the right concrete class for the active
``cfg.llm_provider``. One test per migrated wrapper, parametrised across
every provider we ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _Cfg:
    """Minimal cfg shape the factory reads."""
    llm_provider: str = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""
    llm_chat_model: str = "test-chat"
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "test-embed"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "test-chat"
    ollama_embed_model: str = "test-embed"
    llm_chat_timeout_sec: float = 30.0
    llm_thinking_enabled: bool = False
    gemini_api_key: str = ""
    gemini_base_url: str = ""


def _ollama_cfg() -> _Cfg:
    return _Cfg(llm_provider="ollama")


def _openai_cfg() -> _Cfg:
    return _Cfg(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="sk-test",
    )


def _gemini_cfg() -> _Cfg:
    return _Cfg(llm_provider="gemini", gemini_api_key="g-test")


def _backend_classes():
    from src.jarvis.llm.gemini import GeminiBackend
    from src.jarvis.llm.ollama import OllamaBackend
    from src.jarvis.llm.openai_compatible import OpenAICompatibleBackend

    return {
        "ollama": OllamaBackend,
        "openai_compatible": OpenAICompatibleBackend,
        "gemini": GeminiBackend,
    }


def _patch_all(method: str, return_value):
    """Patch ``method`` on every backend class; returns (stack, mocks by provider)."""
    from contextlib import ExitStack

    stack = ExitStack()
    mocks = {
        name: stack.enter_context(patch.object(cls, method, return_value=return_value(name)))
        for name, cls in _backend_classes().items()
    }
    return stack, mocks


def _assert_only(mocks, expected: str) -> None:
    for name, mock in mocks.items():
        if name == expected:
            assert mock.called, f"expected the {name} backend to be invoked"
        else:
            assert not mock.called, f"the {name} backend must not be invoked for llm_provider={expected}"


# ── direct() wrappers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "module_path, fn_name, cfg_factory, expected_backend_module",
    [
        # Reply path
        ("src.jarvis.reply.planner", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.planner", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.reply.planner", "call_llm_direct", _gemini_cfg, "gemini"),
        ("src.jarvis.reply.evaluator", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.evaluator", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.reply.evaluator", "call_llm_direct", _gemini_cfg, "gemini"),
        ("src.jarvis.reply.enrichment", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.reply.enrichment", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.reply.enrichment", "call_llm_direct", _gemini_cfg, "gemini"),
        # Memory path
        ("src.jarvis.memory.graph_ops", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.memory.graph_ops", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.memory.graph_ops", "call_llm_direct", _gemini_cfg, "gemini"),
        # Builtin tools
        ("src.jarvis.tools.builtin.nutrition.log_meal", "call_llm_direct", _ollama_cfg, "ollama"),
        ("src.jarvis.tools.builtin.nutrition.log_meal", "call_llm_direct", _openai_cfg, "openai_compatible"),
        ("src.jarvis.tools.builtin.nutrition.log_meal", "call_llm_direct", _gemini_cfg, "gemini"),
    ],
)
def test_call_llm_direct_wrapper_dispatches_via_factory(
    module_path: str, fn_name: str, cfg_factory, expected_backend_module: str
):
    """Each module's local ``call_llm_direct`` must route through
    ``get_llm_backend(cfg)`` so swapping ``llm_provider`` swaps the backend."""
    import importlib
    mod = importlib.import_module(module_path)
    wrapper = getattr(mod, fn_name)
    cfg = cfg_factory()

    # Patch the concrete backend classes' .direct so we can see which one
    # the wrapper actually called. Patching at the class level catches the
    # backend regardless of how the factory constructs it.
    stack, mocks = _patch_all("direct", lambda name: f"{name}-result")
    with stack:
        result = wrapper(
            cfg=cfg,
            chat_model=cfg.llm_chat_model,
            system_prompt="sys",
            user_content="user",
            timeout_sec=1.0,
        )

    _assert_only(mocks, expected_backend_module)
    assert result == f"{expected_backend_module}-result"


# ── chat() wrapper (engine) ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cfg_factory, expected_backend_module",
    [
        (_ollama_cfg, "ollama"),
        (_openai_cfg, "openai_compatible"),
        (_gemini_cfg, "gemini"),
    ],
)
def test_engine_chat_with_messages_dispatches_via_factory(cfg_factory, expected_backend_module: str):
    """``engine.chat_with_messages`` (the agentic-loop boundary) must dispatch
    through the factory too — the chat shape is the largest LLM call in the
    app and silently falling back to Ollama would defeat the entire migration."""
    from src.jarvis.reply import engine as engine_mod

    cfg = cfg_factory()

    stack, mocks = _patch_all("chat", lambda name: {"message": {"content": name}})
    with stack:
        engine_mod.chat_with_messages(cfg, [{"role": "user", "content": "hi"}], timeout_sec=1.0)

    _assert_only(mocks, expected_backend_module)


# ── weather extractor (uses get_llm_backend directly, no local wrapper) ────

@pytest.mark.parametrize(
    "cfg_factory, expected_backend_module",
    [
        (_ollama_cfg, "ollama"),
        (_openai_cfg, "openai_compatible"),
        (_gemini_cfg, "gemini"),
    ],
)
def test_weather_place_extractor_dispatches_via_factory(cfg_factory, expected_backend_module: str):
    from src.jarvis.tools.builtin import weather as weather_mod

    cfg = cfg_factory()

    stack, mocks = _patch_all("direct", lambda name: "London")
    with stack:
        weather_mod._extract_place_from_user_text("weather in london please", cfg)

    _assert_only(mocks, expected_backend_module)
