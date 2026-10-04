"""Gemini as an LLM provider: config resolution and factory dispatch.

Pins the mechanism: selecting ``llm_provider: "gemini"`` routes chat, the
fast tier and embeddings to Google's API with Gemini model names, keeps each
provider's model fields from leaking into the others, and lets the API key
come from the config file or the conventional environment variables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _load(tmp_path, monkeypatch, cfg: dict):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"_config_version": 3, **cfg}))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
    from jarvis.config import load_settings
    return load_settings()


class TestGeminiConfigResolution:
    def test_unset_models_resolve_to_gemini_defaults(self, tmp_path, monkeypatch):
        from jarvis.config import (
            DEFAULT_GEMINI_CHAT_MODEL,
            DEFAULT_GEMINI_EMBED_MODEL,
            DEFAULT_GEMINI_FAST_MODEL,
        )

        s = _load(tmp_path, monkeypatch, {"llm_provider": "gemini", "gemini_api_key": "k"})

        assert s.llm_provider == "gemini"
        assert s.llm_chat_model == DEFAULT_GEMINI_CHAT_MODEL
        assert s.fast_model == DEFAULT_GEMINI_FAST_MODEL
        assert s.embedding_model == DEFAULT_GEMINI_EMBED_MODEL

    def test_explicit_gemini_models_win(self, tmp_path, monkeypatch):
        s = _load(tmp_path, monkeypatch, {
            "llm_provider": "gemini",
            "gemini_chat_model": "chat-x",
            "gemini_fast_model": "fast-x",
            "gemini_embed_model": "embed-x",
        })

        assert (s.llm_chat_model, s.fast_model, s.embedding_model) == ("chat-x", "fast-x", "embed-x")

    def test_other_providers_model_fields_do_not_leak_into_gemini(self, tmp_path, monkeypatch):
        s = _load(tmp_path, monkeypatch, {
            "llm_provider": "gemini",
            "ollama_chat_model": "gemma4:e4b",
            "llm_chat_model": "lmstudio-model",
            "fast_model": "qwen3:1.7b",
            "embedding_model": "text-embedding-3-small",
        })

        for value in (s.llm_chat_model, s.fast_model, s.embedding_model):
            assert value not in {"gemma4:e4b", "lmstudio-model", "qwen3:1.7b", "text-embedding-3-small"}

    def test_gemini_fields_do_not_leak_back_into_ollama(self, tmp_path, monkeypatch):
        from jarvis.config import DEFAULT_FAST_MODEL

        s = _load(tmp_path, monkeypatch, {
            "llm_provider": "ollama",
            "ollama_chat_model": "gemma4:e4b",
            "gemini_chat_model": "chat-x",
            "gemini_fast_model": "fast-x",
        })

        assert s.llm_chat_model == "gemma4:e4b"
        assert s.fast_model == DEFAULT_FAST_MODEL

    def test_embeddings_can_stay_local_while_chat_uses_gemini(self, tmp_path, monkeypatch):
        s = _load(tmp_path, monkeypatch, {
            "llm_provider": "gemini",
            "embedding_provider": "ollama",
            "ollama_embed_model": "nomic-embed-text",
        })

        assert s.embedding_model == "nomic-embed-text"

    def test_api_key_from_config(self, tmp_path, monkeypatch):
        s = _load(tmp_path, monkeypatch, {"llm_provider": "gemini", "gemini_api_key": " cfg-key "})

        assert s.gemini_api_key == "cfg-key"

    @pytest.mark.parametrize("env_name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    def test_api_key_falls_back_to_environment(self, tmp_path, monkeypatch, env_name):
        monkeypatch.setenv(env_name, "env-key")

        s = _load(tmp_path, monkeypatch, {"llm_provider": "gemini"})

        assert s.gemini_api_key == "env-key"

    def test_config_key_beats_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")

        s = _load(tmp_path, monkeypatch, {"llm_provider": "gemini", "gemini_api_key": "cfg-key"})

        assert s.gemini_api_key == "cfg-key"


@dataclass
class _Cfg:
    llm_provider: str = "gemini"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_chat_model: str = ""
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "gemma4:e2b"
    ollama_embed_model: str = "nomic-embed-text"
    gemini_api_key: str = "g-key"
    gemini_base_url: str = ""


class TestGeminiFactoryDispatch:
    def test_chat_backend_is_gemini_on_the_official_endpoint(self):
        from jarvis.llm import GeminiBackend, get_llm_backend
        from jarvis.llm.gemini import DEFAULT_GEMINI_BASE_URL

        backend = get_llm_backend(_Cfg())

        assert isinstance(backend, GeminiBackend)
        assert backend.base_url == DEFAULT_GEMINI_BASE_URL

    def test_custom_gemini_endpoint_is_honoured(self):
        from jarvis.llm import get_llm_backend

        backend = get_llm_backend(_Cfg(gemini_base_url="https://gateway.example/v1beta/"))

        assert backend.base_url == "https://gateway.example/v1beta"

    def test_openai_compatible_url_never_leaks_into_gemini(self):
        from jarvis.llm import get_llm_backend
        from jarvis.llm.gemini import DEFAULT_GEMINI_BASE_URL

        backend = get_llm_backend(_Cfg(llm_base_url="http://localhost:1234/v1"))

        assert backend.base_url == DEFAULT_GEMINI_BASE_URL

    def test_embeddings_inherit_gemini(self):
        from jarvis.llm import GeminiBackend, get_embedding_backend

        assert isinstance(get_embedding_backend(_Cfg()), GeminiBackend)

    def test_embeddings_can_be_routed_to_gemini_from_an_ollama_chat(self):
        from jarvis.llm import GeminiBackend, OllamaBackend, get_embedding_backend, get_llm_backend

        cfg = _Cfg(llm_provider="ollama", embedding_provider="gemini")

        assert isinstance(get_llm_backend(cfg), OllamaBackend)
        assert isinstance(get_embedding_backend(cfg), GeminiBackend)

    def test_embeddings_can_stay_on_ollama(self):
        from jarvis.llm import OllamaBackend, get_embedding_backend

        backend = get_embedding_backend(_Cfg(embedding_provider="ollama"))

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_gemini_key_is_sent(self):
        from jarvis.llm import get_llm_backend

        headers = get_llm_backend(_Cfg(gemini_api_key="secret"))._headers()

        assert headers.get("x-goog-api-key") == "secret"
