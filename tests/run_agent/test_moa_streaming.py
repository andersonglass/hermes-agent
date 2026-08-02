"""Tests for MoA aggregator streaming.

MoAChatCompletions.create() honors stream=True by running the references first
and then returning the aggregator's raw streaming iterator (from call_llm), so
the acting model's output can stream to the user. stream=False is the original
complete-response path and must stay byte-identical.
"""
import threading
from types import SimpleNamespace

import pytest


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _write_cfg(home):
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _facade(monkeypatch, tmp_path, on_call=None):
    home = tmp_path / ".hermes"
    _write_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if on_call is not None:
            r = on_call(kwargs)
            if r is not None:
                return r
        if kwargs["task"] == "moa_reference":
            return _response("reference advice")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    from agent.moa_loop import MoAChatCompletions

    return MoAChatCompletions("review"), calls


# --------------------------------------------------------------------------
# Facade-level: create() stream branch
# --------------------------------------------------------------------------

def test_create_streams_aggregator_when_requested(monkeypatch, tmp_path):
    """stream=True: references still run, aggregator is called with stream=True
    and stream_options, and create() returns the aggregator call's result
    (the raw stream) verbatim."""
    sentinel = object()

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return sentinel
        return None

    facade, calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )

    # create() returns the aggregator's streaming result untouched.
    assert out is sentinel
    # References still ran (MoA not bypassed).
    assert any(c["task"] == "moa_reference" for c in calls)
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["stream"] is True
    assert agg["stream_options"] == {"include_usage": True}
    # Tools still flow to the (streaming) aggregator.
    assert agg["tools"] is not None


def test_create_non_stream_path_unchanged(monkeypatch, tmp_path):
    """Default (no stream): the aggregator call carries NO stream/stream_options
    keys, so the non-streaming path is byte-identical to before."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(messages=[{"role": "user", "content": "q"}], tools=[])

    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert "stream" not in agg
    assert "stream_options" not in agg
    assert "timeout" not in agg


def test_create_forwards_stream_read_timeout(monkeypatch, tmp_path):
    """The consumer's per-request (stream read) timeout is forwarded to the
    aggregator so it actually governs the stream."""
    timeout_sentinel = object()
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
        timeout=timeout_sentinel,
    )
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["timeout"] is timeout_sentinel


def test_create_respects_caller_stream_options(monkeypatch, tmp_path):
    """A caller-provided stream_options is forwarded as-is (not overwritten)."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
        stream_options={"include_usage": False, "extra": 1},
    )
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["stream_options"] == {"include_usage": False, "extra": 1}


def test_create_does_not_forward_timeout_when_not_streaming(monkeypatch, tmp_path):
    """A stray timeout on a non-streaming call is NOT forwarded — the non-stream
    path must remain unchanged regardless of incidental kwargs."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(messages=[{"role": "user", "content": "q"}], tools=[], timeout=object())
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert "timeout" not in agg
    assert "stream" not in agg


# --------------------------------------------------------------------------
# call_llm-level: stream branch returns the raw SDK stream
# --------------------------------------------------------------------------

def test_call_llm_stream_returns_raw_stream_and_skips_validation(monkeypatch):
    """call_llm(stream=True) returns the client's raw stream object directly,
    attaches stream/stream_options to the request, and does NOT run response
    validation (which assumes a complete response)."""
    from agent import auxiliary_client as ac

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "RAW_STREAM"

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="http://localhost:8001/v1",
    )

    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("custom", "m", "http://localhost:8001/v1", "key", "chat_completions"),
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "m"))

    def _no_validate(*a, **k):
        raise AssertionError("streaming must not go through _validate_llm_response")

    monkeypatch.setattr(ac, "_validate_llm_response", _no_validate)

    out = ac.call_llm(
        provider="custom",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    assert out == "RAW_STREAM"
    assert captured.get("stream") is True
    assert captured.get("stream_options") == {"include_usage": True}


def test_call_llm_non_stream_still_validates(monkeypatch):
    """Sanity: stream=False keeps the validated path (regression guard for the
    early-return not leaking into normal calls)."""
    from agent import auxiliary_client as ac

    class _Completions:
        def create(self, **kwargs):
            return _response("ok")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="http://localhost:8001/v1",
    )
    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("custom", "m", "http://localhost:8001/v1", "key", "chat_completions"),
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "m"))

    validated = {"called": False}

    def _validate(resp, task, provider=None, base_url=None):
        validated["called"] = True
        return resp

    monkeypatch.setattr(ac, "_validate_llm_response", _validate)

    ac.call_llm(
        provider="custom",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert validated["called"] is True


def test_relay_stream_adapts_complete_response_from_internal_streaming_client(monkeypatch):
    """Adapters such as openai-codex consume their wire stream internally and
    return a complete ChatCompletion even when ``stream=True`` was requested.
    The MoA relay seam must expose that completion as a valid chunk iterator
    instead of attempting ``iter(SimpleNamespace)``.
    """
    from agent import auxiliary_client as ac

    complete = _response("streamed through adapter")

    class _Completions:
        def create(self, **_kwargs):
            return complete

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    monkeypatch.setattr(ac, "_relay_auxiliary_metadata", lambda **_kwargs: None)

    chunks = list(ac._relay_sync_stream(client, {"model": "fake", "stream": True}))

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "streamed through adapter"
    assert chunks[0].choices[0].finish_reason == "stop"


def test_relay_stream_adds_indices_when_adapting_complete_tool_calls(monkeypatch):
    from agent import auxiliary_client as ac

    complete = SimpleNamespace(
        id="complete-1",
        created=1,
        model="fake",
        usage=None,
        choices=[SimpleNamespace(
            index=0,
            finish_reason="tool_calls",
            message=SimpleNamespace(
                role="assistant",
                content=None,
                reasoning=None,
                reasoning_content=None,
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    type="function",
                    function=SimpleNamespace(name="lookup", arguments='{"job":"42"}'),
                )],
            ),
        )],
    )

    class _Completions:
        def create(self, **_kwargs):
            return complete

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    monkeypatch.setattr(ac, "_relay_auxiliary_metadata", lambda **_kwargs: None)

    chunk = list(ac._relay_sync_stream(client, {"model": "fake", "stream": True}))[0]
    tool_delta = chunk.choices[0].delta.tool_calls[0]

    assert tool_delta.index == 0
    assert tool_delta.id == "call-1"
    assert tool_delta.type == "function"
    assert tool_delta.function.name == "lookup"
    assert tool_delta.function.arguments == '{"job":"42"}'


def test_codex_auxiliary_adapter_emits_incremental_text_chunks():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    events = [
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", phase="final"),
        ),
        SimpleNamespace(type="response.output_text.delta", delta="Hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello")],
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp-1",
                status="completed",
                usage=SimpleNamespace(input_tokens=2, output_tokens=2, total_tokens=4),
            ),
        ),
    ]
    captured = {}

    def create_response(**kwargs):
        captured.update(kwargs)
        return iter(events)

    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=create_response),
    )
    adapter = _CodexCompletionsAdapter(real_client, "gpt-test")

    chunks = list(adapter.create(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-test",
        stream=True,
    ))

    assert captured["stream"] is True
    assert [
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices[0].delta.content
    ] == ["Hel", "lo"]
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.total_tokens == 4


def test_codex_auxiliary_adapter_streams_indexed_tool_calls():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="lookup",
        arguments='{"id":42}',
    )
    events = [
        SimpleNamespace(type="response.output_item.added", item=function_call),
        SimpleNamespace(type="response.output_item.done", item=function_call),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp-2", status="completed", usage=None),
        ),
    ]
    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **_: iter(events)),
    )
    adapter = _CodexCompletionsAdapter(real_client, "gpt-test")

    chunks = list(adapter.create(
        messages=[{"role": "user", "content": "use a tool"}],
        model="gpt-test",
        stream=True,
    ))

    terminal = chunks[-1]
    assert terminal.choices[0].finish_reason == "tool_calls"
    tool_delta = terminal.choices[0].delta.tool_calls[0]
    assert tool_delta.index == 0
    assert tool_delta.id == "call-1"
    assert tool_delta.function.name == "lookup"


def test_codex_auxiliary_adapter_close_before_iteration_never_starts_provider():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    provider_calls = []

    def create_response(**kwargs):
        provider_calls.append(kwargs)
        return iter(())

    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=create_response),
    )
    adapter = _CodexCompletionsAdapter(real_client, "gpt-test")

    stream = adapter.create(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-test",
        stream=True,
    )
    stream.close()

    assert provider_calls == []
    assert list(stream) == []


def test_codex_auxiliary_adapter_close_unblocks_provider_publication_race():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    create_entered = threading.Event()
    allow_provider_publication = threading.Event()

    class BlockingEventStream:
        def __init__(self):
            self.close_called = threading.Event()
            self.released = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            self.released.wait(timeout=5)
            raise StopIteration

        def close(self):
            self.close_called.set()
            self.released.set()

    provider_stream = BlockingEventStream()

    def create_response(**_kwargs):
        create_entered.set()
        allow_provider_publication.wait(timeout=5)
        return provider_stream

    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=create_response),
    )
    adapter = _CodexCompletionsAdapter(real_client, "gpt-test")
    stream = adapter.create(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-test",
        stream=True,
    )

    consumer = threading.Thread(target=lambda: next(stream, None))
    consumer.start()
    assert create_entered.wait(timeout=1)
    producer = stream._producer

    stream.close()
    allow_provider_publication.set()
    provider_was_closed = provider_stream.close_called.wait(timeout=1)
    producer.join(timeout=1)
    consumer.join(timeout=1)
    producer_stopped = not producer.is_alive()
    consumer_stopped = not consumer.is_alive()

    # Always release a broken implementation so this regression test cannot
    # strand a daemon producer after its expected RED failure.
    provider_stream.released.set()
    producer.join(timeout=1)
    consumer.join(timeout=1)

    assert provider_was_closed
    assert producer_stopped
    assert consumer_stopped


def test_codex_auxiliary_adapter_close_interrupts_started_provider_stream():
    from agent.auxiliary_client import _CodexCompletionsAdapter

    class BlockingEventStream:
        def __init__(self):
            self.event_index = 0
            self.close_called = threading.Event()
            self.released = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            if self.event_index == 0:
                self.event_index += 1
                return SimpleNamespace(type="response.output_text.delta", delta="first")
            self.released.wait(timeout=5)
            raise StopIteration

        def close(self):
            self.close_called.set()
            self.released.set()

    provider_stream = BlockingEventStream()
    real_client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=lambda **_: provider_stream),
    )
    adapter = _CodexCompletionsAdapter(real_client, "gpt-test")
    stream = adapter.create(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-test",
        stream=True,
    )

    assert next(stream).choices[0].delta.content == "first"
    producer = stream._producer
    stream.close()

    assert provider_stream.close_called.wait(timeout=1)
    producer.join(timeout=1)
    assert not producer.is_alive()
    assert list(stream) == []
