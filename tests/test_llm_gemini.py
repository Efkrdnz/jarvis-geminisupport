"""Behaviour tests for the native Google Gemini backend.

The backend is exercised against a small in-process HTTP server that speaks
the Gemini REST shape (``models/{model}:generateContent``,
``:streamGenerateContent?alt=sse``, ``:embedContent``, ``GET models``), so the
tests observe exactly what goes over the wire and what the reply engine gets
back, without patching the backend's internals.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import pytest
import requests

pytestmark = pytest.mark.unit


API_KEY = "test-gemini-key-123"


class FakeGemini:
    """Scripted Gemini REST server. Each request is recorded; responses are
    popped from a per-route queue, falling back to a per-route default."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.queued: Dict[str, List[tuple]] = {}
        self.defaults: Dict[str, tuple] = {}
        # Optional ``(route, body) -> (status, body)`` hook that answers
        # based on the request itself (used by the engine round trip).
        self.responder = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def _route(self) -> str:
                path = urlparse(self.path).path
                if path.endswith(":generateContent"):
                    return "generate"
                if path.endswith(":streamGenerateContent"):
                    return "stream"
                if path.endswith(":embedContent"):
                    return "embed"
                if path.rstrip("/").endswith("/models"):
                    return "list"
                return "model"

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                route = self._route()
                fake.requests.append({
                    "method": method,
                    "route": route,
                    "path": urlparse(self.path).path,
                    "query": parse_qs(urlparse(self.path).query),
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(raw) if raw else None,
                })
                queue = fake.queued.get(route) or []
                if fake.responder is not None and not queue:
                    status, body = fake.responder(route, fake.requests[-1]["body"])
                elif queue:
                    status, body = queue.pop(0)
                else:
                    status, body = fake.defaults.get(route, (404, {"error": {"message": "no route"}}))
                if route == "stream" and status == 200:
                    payload = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in body)
                    data = payload.encode()
                    content_type = "text/event-stream"
                else:
                    data = json.dumps(body).encode()
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1beta"

    def queue(self, route: str, status: int, body: Any) -> None:
        self.queued.setdefault(route, []).append((status, body))

    def bodies(self, route: str) -> List[Dict[str, Any]]:
        return [r["body"] for r in self.requests if r["route"] == route]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    server = FakeGemini()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _fresh_thinking_cache():
    from jarvis.llm import gemini

    gemini.reset_model_quirks()
    yield
    gemini.reset_model_quirks()


def _backend(fake: FakeGemini):
    from jarvis.llm import GeminiBackend

    return GeminiBackend(fake.base_url, api_key=API_KEY)


def _text_reply(text: str, **extra) -> Dict[str, Any]:
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text, **extra}]},
            "finishReason": "STOP",
        }]
    }


def _closed_port_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}/v1beta"


# ---------------------------------------------------------------------------
# direct()
# ---------------------------------------------------------------------------


class TestDirect:
    def test_returns_reply_text(self, fake):
        fake.queue("generate", 200, _text_reply("hello there"))

        assert _backend(fake).direct("gemini-flash-latest", "sys", "hi") == "hello there"

    def test_targets_the_model_generate_endpoint(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "sys", "hi")

        assert fake.requests[0]["path"] == "/v1beta/models/gemini-flash-latest:generateContent"

    def test_api_key_travels_in_header_not_url(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "sys", "hi")

        req = fake.requests[0]
        assert req["headers"].get("x-goog-api-key") == API_KEY
        assert API_KEY not in req["path"]
        assert all(API_KEY not in v for vals in req["query"].values() for v in vals)

    def test_system_prompt_becomes_system_instruction(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "be brief", "hi")

        body = fake.bodies("generate")[0]
        assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
        assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]

    def test_sampling_controls_map_to_generation_config(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "s", "u", temperature=0.2, max_tokens=50)

        gen = fake.bodies("generate")[0]["generationConfig"]
        assert gen["temperature"] == 0.2
        assert gen["maxOutputTokens"] == 50

    def test_thought_parts_are_not_returned_as_reply_text(self, fake):
        fake.queue("generate", 200, {
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "let me think", "thought": True},
                {"text": "final answer"},
            ]}}]
        })

        assert _backend(fake).direct("gemini-flash-latest", "s", "u") == "final answer"

    def test_http_error_returns_none_without_leaking_key(self, fake, capsys):
        fake.queue("generate", 403, {"error": {"message": f"bad key {API_KEY}"}})

        assert _backend(fake).direct("gemini-flash-latest", "s", "u") is None
        out = capsys.readouterr()
        assert API_KEY not in out.out + out.err

    def test_blocked_or_empty_candidate_returns_none(self, fake):
        fake.queue("generate", 200, {"promptFeedback": {"blockReason": "SAFETY"}})

        assert _backend(fake).direct("gemini-flash-latest", "s", "u") is None


# ---------------------------------------------------------------------------
# Thinking control — latency is the point: classification passes must not
# pay for a full reasoning trace.
# ---------------------------------------------------------------------------


class TestThinking:
    def test_thinking_off_uses_the_cheapest_thinking_level(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "s", "u", thinking=False)

        gen = fake.bodies("generate")[0]["generationConfig"]
        assert gen["thinkingConfig"] == {"thinkingLevel": "minimal"}

    def test_thinking_off_on_budget_era_models_uses_zero_budget(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-2.5-flash", "s", "u", thinking=False)

        gen = fake.bodies("generate")[0]["generationConfig"]
        assert gen["thinkingConfig"] == {"thinkingBudget": 0}

    def test_thinking_on_leaves_the_model_default(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).direct("gemini-flash-latest", "s", "u", thinking=True)

        gen = fake.bodies("generate")[0].get("generationConfig", {})
        assert "thinkingConfig" not in gen

    def test_rejected_thinking_config_falls_back_and_still_answers(self, fake):
        fake.queue("generate", 400, {"error": {"message": "unsupported thinking level"}})
        fake.queue("generate", 200, _text_reply("recovered"))

        assert _backend(fake).direct("gemini-pro-latest", "s", "u") == "recovered"
        first, second = fake.bodies("generate")
        assert first["generationConfig"]["thinkingConfig"] != second["generationConfig"].get("thinkingConfig")

    def test_working_thinking_config_is_remembered_across_calls(self, fake):
        fake.queue("generate", 400, {"error": {"message": "unsupported thinking level"}})
        fake.queue("generate", 200, _text_reply("one"))
        fake.queue("generate", 200, _text_reply("two"))
        backend = _backend(fake)

        backend.direct("gemini-pro-latest", "s", "u")
        before = len(fake.requests)
        assert backend.direct("gemini-pro-latest", "s", "u") == "two"

        assert len(fake.requests) - before == 1, "second call should not re-probe a rejected config"

    def test_model_that_rejects_all_thinking_controls_runs_without_them(self, fake):
        for _ in range(2):
            fake.queue("generate", 400, {"error": {"message": "thinking not supported"}})
        fake.queue("generate", 200, _text_reply("plain"))

        assert _backend(fake).direct("gemma-3-27b-it", "s", "u") == "plain"
        last = fake.bodies("generate")[-1]
        assert "thinkingConfig" not in last.get("generationConfig", {})


# ---------------------------------------------------------------------------
# chat() — tools, thought signatures, message translation
# ---------------------------------------------------------------------------


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "getWeather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


BAD_KEY_ERROR = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [{
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": "API_KEY_INVALID",
            "domain": "googleapis.com",
        }],
    }
}


def _call_reply(name: str, args: Dict[str, Any], signature: Optional[str] = "sig-abc") -> Dict[str, Any]:
    part: Dict[str, Any] = {"functionCall": {"name": name, "args": args}}
    if signature:
        part["thoughtSignature"] = signature
    return {"candidates": [{"content": {"role": "model", "parts": [part]}}]}


class TestChatTools:
    def test_tools_are_declared_as_function_declarations(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))

        _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])

        decls = fake.bodies("generate")[0]["tools"][0]["functionDeclarations"]
        assert decls == [{
            "name": "getWeather",
            "description": "Get the weather",
            "parametersJsonSchema": WEATHER_TOOL["function"]["parameters"],
        }]

    def test_function_call_is_normalised_to_engine_tool_calls(self, fake):
        fake.queue("generate", 200, _call_reply("getWeather", {"city": "Paris"}))

        resp = _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL])

        calls = resp["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"] == {"name": "getWeather", "arguments": {"city": "Paris"}}
        assert calls[0]["id"], "every tool call needs an id so tool results can be paired"

    def test_thought_signature_round_trips_to_the_next_request(self, fake):
        fake.queue("generate", 200, _call_reply("getWeather", {"city": "Paris"}, signature="sig-xyz"))
        fake.queue("generate", 200, _text_reply("It is sunny."))
        backend = _backend(fake)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "weather?"}]

        resp = backend.chat("gemini-flash-latest", messages, tools=[WEATHER_TOOL])
        call = resp["message"]["tool_calls"][0]
        # Mirror what the reply engine appends to the conversation.
        messages.append({"role": "assistant", "content": "", "tool_calls": resp["message"]["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "sunny, 21C", "tool_name": "getWeather"})
        final = backend.chat("gemini-flash-latest", messages, tools=[WEATHER_TOOL])

        assert final["message"]["content"] == "It is sunny."
        contents = fake.bodies("generate")[1]["contents"]
        model_turn = contents[1]
        assert model_turn["role"] == "model"
        assert model_turn["parts"][0]["functionCall"] == {"name": "getWeather", "args": {"city": "Paris"}}
        assert model_turn["parts"][0]["thoughtSignature"] == "sig-xyz"
        result_turn = contents[2]
        assert result_turn["role"] == "user"
        response_part = result_turn["parts"][0]["functionResponse"]
        assert response_part["name"] == "getWeather"
        assert "sunny, 21C" in json.dumps(response_part["response"])

    def test_history_tool_calls_without_signature_still_validate(self, fake):
        from jarvis.llm.gemini import SKIP_THOUGHT_SIGNATURE

        fake.queue("generate", 200, _text_reply("ok"))
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "getWeather", "arguments": {"city": "Rome"}}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "rainy"},
        ]

        _backend(fake).chat("gemini-flash-latest", messages, tools=[WEATHER_TOOL])

        model_turn = fake.bodies("generate")[0]["contents"][1]
        assert model_turn["parts"][0]["thoughtSignature"] == SKIP_THOUGHT_SIGNATURE

    def test_unanswered_parallel_calls_are_not_replayed(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))
        messages = [
            {"role": "user", "content": "weather in two cities?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "getWeather", "arguments": {"city": "Rome"}}, "thought_signature": "s1"},
                {"id": "b", "type": "function", "function": {"name": "getWeather", "arguments": {"city": "Oslo"}}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "rainy"},
        ]

        _backend(fake).chat("gemini-flash-latest", messages, tools=[WEATHER_TOOL])

        contents = fake.bodies("generate")[0]["contents"]
        calls = [p for p in contents[1]["parts"] if "functionCall" in p]
        responses = [p for p in contents[2]["parts"] if "functionResponse" in p]
        assert len(calls) == len(responses) == 1
        assert calls[0]["functionCall"]["args"] == {"city": "Rome"}

    def test_tool_rejection_raises_tools_not_supported(self, fake):
        from jarvis.llm import ToolsNotSupportedError

        fake.defaults["generate"] = (400, {"error": {"message": "function calling unsupported"}})

        with pytest.raises(ToolsNotSupportedError):
            _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])

    def test_connection_failure_is_re_raised(self):
        from jarvis.llm import GeminiBackend

        backend = GeminiBackend(_closed_port_url(), api_key=API_KEY)

        with pytest.raises(requests.ConnectionError):
            backend.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])

    def test_invalid_key_is_not_mistaken_for_missing_tool_support(self, fake, capsys):
        fake.defaults["generate"] = (400, BAD_KEY_ERROR)

        resp = _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL])

        assert resp is None
        assert len(fake.bodies("generate")) == 1, "a bad key must not trigger thinking-control retries"
        assert "API key" in capsys.readouterr().out

    def test_rate_limit_fails_soft(self, fake):
        fake.queue("generate", 429, {"error": {"message": "quota"}})

        assert _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "hi"}]) is None


class TestMessageTranslation:
    def test_later_system_messages_stay_in_conversation_order(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))
        messages = [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "nudge: be concise"},
            {"role": "user", "content": "and now?"},
        ]

        _backend(fake).chat("gemini-flash-latest", messages)

        body = fake.bodies("generate")[0]
        assert body["systemInstruction"]["parts"][0]["text"] == "persona"
        roles = [c["role"] for c in body["contents"]]
        assert roles == ["user", "model", "user"]
        last_texts = [p["text"] for p in body["contents"][2]["parts"]]
        assert any("nudge: be concise" in t for t in last_texts)
        assert any("and now?" in t for t in last_texts)

    def test_engine_internal_fields_never_reach_the_wire(self, fake):
        fake.queue("generate", 200, _text_reply("ok"))
        messages = [
            {"role": "system", "content": "s", "_is_context_injected": True},
            {"role": "user", "content": "hi", "tool_name": "x"},
        ]

        _backend(fake).chat("gemini-flash-latest", messages)

        raw = json.dumps(fake.bodies("generate")[0])
        assert "_is_context_injected" not in raw
        assert "tool_name" not in raw

    def test_json_format_requests_json_mime_type(self, fake):
        fake.queue("generate", 200, _text_reply("{}"))

        _backend(fake).chat("gemini-flash-latest", [{"role": "user", "content": "hi"}],
                            extra_options={"format": "json", "keep_alive": "5m", "num_ctx": 8192})

        body = fake.bodies("generate")[0]
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        raw = json.dumps(body)
        assert "keep_alive" not in raw and "num_ctx" not in raw


# ---------------------------------------------------------------------------
# streaming()
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_streams_tokens_and_returns_full_text(self, fake):
        fake.queue("stream", 200, [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "pondering", "thought": True}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}}]},
        ])
        tokens: List[str] = []

        result = _backend(fake).streaming("gemini-flash-latest", "s", "u", on_token=tokens.append)

        assert result == "Hello"
        assert tokens == ["Hel", "lo"]
        assert fake.requests[0]["query"].get("alt") == ["sse"]


# ---------------------------------------------------------------------------
# embed() / list_models() / warm_up() / check_capabilities()
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_embedding_matches_the_memory_index_dimension(self, fake):
        from jarvis.utils.vector_store import MEMORY_EMBEDDING_DIMENSION

        fake.queue("embed", 200, {"embedding": {"values": [0.1] * MEMORY_EMBEDDING_DIMENSION}})

        vec = _backend(fake).embed("hello", "gemini-embedding-001")

        assert vec is not None and len(vec) == MEMORY_EMBEDDING_DIMENSION
        body = fake.bodies("embed")[0]
        assert body["outputDimensionality"] == MEMORY_EMBEDDING_DIMENSION
        assert body["content"]["parts"][0]["text"] == "hello"
        assert fake.requests[0]["path"] == "/v1beta/models/gemini-embedding-001:embedContent"

    def test_embedding_failure_returns_none(self, fake):
        fake.queue("embed", 500, {"error": {"message": "boom"}})

        assert _backend(fake).embed("hello", "gemini-embedding-001") is None


class TestModelListing:
    def test_lists_generation_models_by_bare_name_across_pages(self, fake):
        fake.queue("list", 200, {
            "models": [
                {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
            ],
            "nextPageToken": "p2",
        })
        fake.queue("list", 200, {
            "models": [{"name": "models/gemini-flash-lite-latest", "supportedGenerationMethods": ["generateContent"]}],
        })

        names = _backend(fake).list_models()

        assert names == ["gemini-flash-latest", "gemini-flash-lite-latest"]
        assert fake.requests[1]["query"].get("pageToken") == ["p2"]

    def test_listing_failure_returns_empty(self, fake):
        fake.queue("list", 403, {"error": {"message": "bad key"}})

        assert _backend(fake).list_models() == []


class TestWarmUp:
    def test_warm_up_succeeds_for_a_reachable_model(self, fake):
        fake.queue("model", 200, {"name": "models/gemini-flash-latest"})

        assert _backend(fake).warm_up("gemini-flash-latest", timeout_sec=5) is True

    def test_warm_up_reports_a_bad_key_or_model(self, fake):
        fake.queue("model", 403, {"error": {"message": "denied"}})

        assert _backend(fake).warm_up("gemini-flash-latest", timeout_sec=5) is False

    def test_warm_up_costs_no_generation(self, fake):
        fake.queue("model", 200, {"name": "models/gemini-flash-latest"})

        _backend(fake).warm_up("gemini-flash-latest", timeout_sec=5)

        assert not fake.bodies("generate"), "cloud warm-up must not spend tokens"


class TestCapabilities:
    def test_reports_chat_tools_and_embeddings(self, fake):
        from jarvis.utils.vector_store import MEMORY_EMBEDDING_DIMENSION

        fake.queue("list", 200, {"models": [
            {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]},
        ]})
        fake.defaults["generate"] = (200, _text_reply("pong"))
        fake.queue("embed", 200, {"embedding": {"values": [0.5] * MEMORY_EMBEDDING_DIMENSION}})

        caps = _backend(fake).check_capabilities("gemini-flash-latest", "gemini-embedding-001")

        assert caps.reachable and caps.chat and caps.tools and caps.embeddings
        assert "gemini-flash-latest" in caps.models


# ---------------------------------------------------------------------------
# Reply engine round trip: the real engine, the real backend, a fake Gemini.
# ---------------------------------------------------------------------------


class TestReplyEngineRoundTrip:
    def test_engine_runs_a_gemini_tool_call_and_answers(self, fake, mock_config, db, dialogue_memory):
        from jarvis.reply import engine as engine_mod
        from jarvis.tools.types import ToolExecutionResult
        from unittest.mock import patch

        mock_config.llm_provider = "gemini"
        mock_config.gemini_base_url = fake.base_url
        mock_config.gemini_api_key = API_KEY
        mock_config.llm_chat_model = "gemini-flash-latest"
        mock_config.fast_model = "gemini-flash-lite-latest"

        def respond(route, body):
            if route != "generate":
                return 404, {"error": {"message": "unused"}}
            parts = [p for c in body.get("contents", []) for p in c.get("parts", [])]
            if any("functionResponse" in p for p in parts):
                return 200, _text_reply("It is 21C and sunny in Paris.")
            if body.get("tools"):
                return 200, _call_reply("getWeather", {"location": "Paris"}, signature="sig-e2e")
            return 200, _text_reply("Check the weather for Paris.")

        fake.responder = respond
        invoked: List[tuple] = []

        def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
            invoked.append((tool_name, tool_args))
            return ToolExecutionResult(success=True, reply_text="Paris: 21C sunny", error_message=None)

        with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
             patch.object(engine_mod, "select_tools", return_value=["getWeather", "stop"]), \
             patch.object(engine_mod, "extract_search_params_for_memory", return_value={"keywords": []}):
            reply = engine_mod.run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text="what's the weather in paris?", dialogue_memory=dialogue_memory,
            )

        assert ("getWeather", {"location": "Paris"}) in invoked
        assert reply and "Paris" in reply
        follow_ups = [
            b for b in fake.bodies("generate")
            if any("functionResponse" in p for c in b.get("contents", []) for p in c.get("parts", []))
        ]
        assert follow_ups, "the tool result must be sent back to Gemini"
        echoed = [
            p.get("thoughtSignature") for c in follow_ups[0]["contents"] for p in c["parts"]
            if "functionCall" in p
        ]
        assert echoed == ["sig-e2e"]
        assert all(r["headers"].get("x-goog-api-key") == API_KEY for r in fake.requests)
