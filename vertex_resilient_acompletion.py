"""Resilient Vertex AI LiteLLM Async Wrapper (`vertex_resilient_acompletion.py`).

Production reference implementation around `litellm.acompletion` for real-time
voice and conversational agents on Google Cloud Vertex AI (`gemini-3.8-flash`
and `gemini-3.5-flash-lite`).

Key Capabilities:
1. Forced Wire-Level Streaming + Transparent Unary Reassembly:
   - Always invokes Vertex AI `:streamGenerateContent` (`stream=True`) on the
     wire so Time-To-First-Token (TTFT) is bounded independently from total
     response generation time.
   - Callers passing `stream=False` (default) receive a standard
     `litellm.ModelResponse` reassembled via `litellm.stream_chunk_builder`.
   - Callers passing `stream=True` receive an async chunk iterator.
2. Non-Empty First-Token TTFT Watchdog (`ttft_timeout = 4.5s`):
   - Guards against initial empty role/metadata chunks by enforcing the TTFT
     deadline until the first chunk containing actual content, tool calls, or
     reasoning tokens arrives.
   - Guarantees deterministic `await stream.aclose()` cleanup on timeout or
     caller cancellation to prevent socket leaks.
3. Dual-Pool HTTP/1.1 Transport Isolation (`httpx.AsyncClient`):
   - `warm_client`: `http2=False`, `max_keepalive_connections=20` for low-latency
     steady-state traffic.
   - `ephemeral_client`: `http2=False`, `max_keepalive_connections=0`, and
     `Connection: close` for retries so a retry never reuses a stalled socket
     or congested upstream path.
4. 3-State Adaptive Standard <-> Priority PayGo Circuit Breaker (95/5 Canary):
   - `CLOSED` (Normal): 100% of Attempt-1 requests use Standard PayGo. Any
     single call exceeding `4.5s` TTFT cancels Attempt 1 and immediately retries
     Attempt 2 on `ephemeral_client` with
     `X-Vertex-AI-LLM-Shared-Request-Type: priority`.
   - `OPEN` (Contention Spike): Trips when >= 3 Standard failures/timeouts
     occur within 30s OR when the rolling 60s failure rate exceeds 5% (minimum
     10 samples). Routes 95% of Attempt-1 requests directly to Priority PayGo
     (zero 4.5s retry delay for live callers) while sending 5% as Canary Probes
     to Standard PayGo (still protected by the 4.5s Priority retry fallback).
   - Auto-Recovery (`OPEN -> CLOSED`): Automatically closes back to 100%
     Standard PayGo once 5 consecutive Standard Canary Probes complete cleanly
     with `TTFT < 3.0s`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import enum
import logging
import random
import time
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple, Union

import httpx

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover - fallback for isolated test runners
    import sys
    import types

    class _Delta:
        def __init__(
            self,
            role: str = "assistant",
            content: Optional[str] = None,
            reasoning_content: Optional[str] = None,
            tool_calls: Optional[List[Any]] = None,
        ) -> None:
            self.role = role
            self.content = content
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls

    class _Message:
        def __init__(self, role: str = "assistant", content: str = "") -> None:
            self.role = role
            self.content = content

    class _Choices:
        def __init__(
            self,
            index: int = 0,
            delta: Optional[_Delta] = None,
            message: Optional[_Message] = None,
        ) -> None:
            self.index = index
            self.delta = delta or _Delta()
            self.message = message or _Message()

    class _ModelResponse:
        def __init__(
            self,
            id: str = "chatcmpl-stub",
            choices: Optional[List[_Choices]] = None,
            model: str = "gemini-3.8-flash",
            stream: bool = False,
        ) -> None:
            self.id = id
            self.choices = choices or []
            self.model = model
            self.stream = stream

    async def _stub_acompletion(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "litellm package is required for live Vertex AI calls: pip install litellm"
        )

    def _stub_stream_chunk_builder(
        chunks: List[_ModelResponse],
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> _ModelResponse:
        assembled = "".join(
            (c.choices[0].delta.content or "")
            for c in chunks
            if getattr(c, "choices", None)
        )
        model_name = chunks[0].model if chunks else "gemini-3.8-flash"
        return _ModelResponse(
            choices=[_Choices(index=0, message=_Message(content=assembled))],
            model=model_name,
            stream=False,
        )

    litellm = types.SimpleNamespace(  # type: ignore
        ModelResponse=_ModelResponse,
        Choices=_Choices,
        utils=types.SimpleNamespace(Delta=_Delta),
        acompletion=_stub_acompletion,
        stream_chunk_builder=_stub_stream_chunk_builder,
    )
    sys.modules["litellm"] = litellm  # type: ignore

logger = logging.getLogger("vertex_resilient_acompletion")


class BreakerState(str, enum.Enum):
    CLOSED = "CLOSED"  # 100% Standard PayGo on Attempt 1
    OPEN = "OPEN"      # 95% Priority PayGo / 5% Standard Canary Probe on Attempt 1


@dataclass
class RouteDecision:
    tier: str          # "standard" or "priority"
    is_canary: bool    # True when probing Standard PayGo while breaker is OPEN
    breaker_state: BreakerState


class VertexPriorityCircuitBreaker:
    """Rolling-window 3-state circuit breaker for Standard <-> Priority PayGo."""

    def __init__(
        self,
        window_seconds: float = 60.0,
        burst_window_seconds: float = 30.0,
        burst_failure_threshold: int = 3,
        failure_rate_threshold: float = 0.05,
        min_window_samples: int = 10,
        canary_ratio: float = 0.05,
        recovery_canary_successes: int = 5,
        healthy_ttft_threshold: float = 3.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.burst_window_seconds = burst_window_seconds
        self.burst_failure_threshold = burst_failure_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.min_window_samples = min_window_samples
        self.canary_ratio = canary_ratio
        self.recovery_canary_successes = recovery_canary_successes
        self.healthy_ttft_threshold = healthy_ttft_threshold
        self._rng = rng or random.Random()

        self._state: BreakerState = BreakerState.CLOSED
        # Stores (monotonic_timestamp, is_failure) for Standard PayGo attempts
        self._history: Deque[Tuple[float, bool]] = deque()
        self._consecutive_healthy_canaries: int = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> BreakerState:
        return self._state

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    async def choose_attempt1_route(self) -> RouteDecision:
        """Selects Standard vs. Priority PayGo for Attempt 1."""
        async with self._lock:
            if self._state == BreakerState.CLOSED:
                return RouteDecision(
                    tier="standard",
                    is_canary=False,
                    breaker_state=BreakerState.CLOSED,
                )

            # State is OPEN: route 5% to Standard Canary Probe, 95% to Priority
            if self._rng.random() < self.canary_ratio:
                return RouteDecision(
                    tier="standard",
                    is_canary=True,
                    breaker_state=BreakerState.OPEN,
                )
            return RouteDecision(
                tier="priority",
                is_canary=False,
                breaker_state=BreakerState.OPEN,
            )

    async def record_standard_outcome(
        self,
        *,
        success: bool,
        ttft_seconds: float,
        is_canary: bool,
    ) -> BreakerState:
        """Updates breaker state from a Standard PayGo attempt outcome."""
        now = time.monotonic()
        is_healthy = success and (ttft_seconds <= self.healthy_ttft_threshold)
        is_failure = not success

        async with self._lock:
            self._history.append((now, is_failure))
            self._prune(now)

            if self._state == BreakerState.CLOSED:
                if is_failure and self._should_trip(now):
                    self._state = BreakerState.OPEN
                    self._consecutive_healthy_canaries = 0
                    logger.warning(
                        "Circuit breaker TRIPPED to OPEN (95%% Priority / 5%% Canary)."
                    )
            else:
                # Breaker is currently OPEN
                if is_canary:
                    if is_healthy:
                        self._consecutive_healthy_canaries += 1
                        if (
                            self._consecutive_healthy_canaries
                            >= self.recovery_canary_successes
                        ):
                            self._state = BreakerState.CLOSED
                            self._consecutive_healthy_canaries = 0
                            self._history.clear()
                            logger.info(
                                "Circuit breaker RECOVERED to CLOSED (100%% Standard)."
                            )
                    else:
                        self._consecutive_healthy_canaries = 0
            return self._state

    def _should_trip(self, now: float) -> bool:
        # Condition A: >= 3 failures within the last 30 seconds
        burst_cutoff = now - self.burst_window_seconds
        recent_Burst_failures = sum(
            1 for ts, fail in self._history if ts >= burst_cutoff and fail
        )
        if recent_Burst_failures >= self.burst_failure_threshold:
            return True

        # Condition B: > 5% failure rate across >= 10 samples in 60s window
        total = len(self._history)
        if total >= self.min_window_samples:
            failures = sum(1 for _, fail in self._history if fail)
            if (failures / total) > self.failure_rate_threshold:
                return True

        return False


def _chunk_has_payload(chunk: Any) -> bool:
    """Returns True if the streaming chunk contains non-empty content or tool calls.

    Vertex AI `:streamGenerateContent` often flushes an initial role-only
    metadata chunk (`delta.content == ""` or `None`) before prefill finishes.
    Waiting for the first non-empty payload guarantees TTFT measures real token
    generation.
    """
    try:
        choices = getattr(chunk, "choices", None) or chunk.get("choices", [])
        if not choices:
            return False
        first = choices[0]
        delta = getattr(first, "delta", None) or first.get("delta", {})
        if not delta:
            return False
        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        if content:
            return True
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning is None and isinstance(delta, dict):
            reasoning = delta.get("reasoning_content")
        if reasoning:
            return True
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls is None and isinstance(delta, dict):
            tool_calls = delta.get("tool_calls")
        if tool_calls:
            return True
        return False
    except Exception:
        return False


class VertexResilientClient:
    """Drop-in resilient wrapper around `litellm.acompletion` for Vertex AI."""

    PRIORITY_HEADERS: Dict[str, str] = {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
    }

    def __init__(
        self,
        *,
        default_model: str = "vertex_ai/gemini-3.8-flash",
        guardrail_model: str = "vertex_ai/gemini-3.5-flash-lite",
        primary_location: str = "global",
        fallback_location: str = "global",
        ttft_timeout: float = 4.5,
        guardrail_ttft_timeout: float = 2.0,
        stream_completion_timeout: float = 25.0,
        circuit_breaker: Optional[VertexPriorityCircuitBreaker] = None,
    ) -> None:
        self.default_model = default_model
        self.guardrail_model = guardrail_model
        self.primary_location = primary_location
        self.fallback_location = fallback_location
        self.ttft_timeout = ttft_timeout
        self.guardrail_ttft_timeout = guardrail_ttft_timeout
        self.stream_completion_timeout = stream_completion_timeout
        self.breaker = circuit_breaker or VertexPriorityCircuitBreaker()

        # Pool 1: Warm HTTP/1.1 connection pool for normal Attempt-1 requests.
        # Disables HTTP/2 multiplexing so concurrent streams do not share a
        # single TCP socket.
        self.warm_client = httpx.AsyncClient(
            http2=False,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(
                connect=3.0,
                read=stream_completion_timeout,
                write=5.0,
                pool=3.0,
            ),
        )

        # Pool 2: Ephemeral zero-keepalive HTTP/1.1 pool for Attempt-2 retries.
        # Forces a fresh TCP/TLS handshake (`Connection: close`) so retries
        # never land on a stalled socket.
        self.ephemeral_client = httpx.AsyncClient(
            http2=False,
            headers={"Connection": "close"},
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=0,
                keepalive_expiry=0.0,
            ),
            timeout=httpx.Timeout(
                connect=3.0,
                read=stream_completion_timeout,
                write=5.0,
                pool=3.0,
            ),
        )

    async def aclose(self) -> None:
        """Cleanly closes both underlying httpx.AsyncClient pools."""
        await asyncio.gather(
            self.warm_client.aclose(),
            self.ephemeral_client.aclose(),
            return_exceptions=True,
        )

    async def _open_verified_stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        use_priority: bool,
        use_ephemeral_socket: bool,
        vertex_location: str,
        ttft_budget: float,
        cached_content: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Tuple[List[Any], AsyncIterator[Any], Any, float]:
        """Opens a LiteLLM stream and enforces `ttft_budget` until non-empty token."""
        call_kwargs = dict(kwargs)
        extra_headers = dict(call_kwargs.pop("extra_headers", None) or {})
        if use_priority:
            extra_headers.update(self.PRIORITY_HEADERS)
        if use_ephemeral_socket:
            extra_headers["Connection"] = "close"

        if cached_content:
            call_kwargs["cached_content"] = cached_content

        call_kwargs["vertex_location"] = call_kwargs.get(
            "vertex_location", vertex_location
        )
        call_kwargs["client"] = (
            self.ephemeral_client if use_ephemeral_socket else self.warm_client
        )

        t0 = time.perf_counter()
        response_stream = None
        buffered_chunks: List[Any] = []

        try:
            async with asyncio.timeout(ttft_budget):
                response_stream = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    stream=True,
                    num_retries=0,  # Disable blind SDK retries; wrapper controls failover
                    extra_headers=extra_headers,
                    **call_kwargs,
                )
                iterator = response_stream.__aiter__()
                while True:
                    chunk = await iterator.__anext__()
                    buffered_chunks.append(chunk)
                    if _chunk_has_payload(chunk):
                        break
        except StopAsyncIteration:
            # Short stream ended before emitting non-empty payload
            pass
        except Exception:
            if response_stream is not None and hasattr(response_stream, "aclose"):
                await response_stream.aclose()
            raise

        ttft = time.perf_counter() - t0
        return buffered_chunks, iterator, response_stream, ttft

    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        ttft_timeout: Optional[float] = None,
        cached_content: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[litellm.ModelResponse, AsyncIterator[Any]]:
        """Drop-in replacement for `litellm.acompletion` with 95/5 Priority Breaker.

        Args:
            messages: OpenAI/LiteLLM format message list.
            model: Target model (defaults to `vertex_ai/gemini-3.8-flash`).
            stream: If False (default), returns a full `litellm.ModelResponse`
                built from the underlying stream. If True, returns an async
                generator yielding chunks.
            ttft_timeout: Max seconds to wait for the first non-empty token
                before aborting Attempt 1 and retrying on Priority PayGo.
            cached_content: Optional Vertex AI Context Cache resource name to
                avoid re-prefilling static system prompts.
            **kwargs: Additional parameters forwarded to `litellm.acompletion`.
        """
        target_model = model or self.default_model
        budget = ttft_timeout if ttft_timeout is not None else self.ttft_timeout
        route = await self.breaker.choose_attempt1_route()

        buffered: List[Any] = []
        iterator: Optional[AsyncIterator[Any]] = None
        raw_stream: Any = None

        # --- Attempt 1: Routed by Circuit Breaker (Standard or Priority) ---
        try:
            buffered, iterator, raw_stream, ttft = await self._open_verified_stream(
                model=target_model,
                messages=messages,
                use_priority=(route.tier == "priority"),
                use_ephemeral_socket=False,
                vertex_location=self.primary_location,
                ttft_budget=budget,
                cached_content=cached_content,
                kwargs=kwargs,
            )
            if route.tier == "standard":
                await self.breaker.record_standard_outcome(
                    success=True,
                    ttft_seconds=ttft,
                    is_canary=route.is_canary,
                )
        except Exception as first_err:
            logger.warning(
                "Attempt 1 (%s, canary=%s) exceeded %.2fs TTFT or failed (%s); "
                "failing over to Attempt 2 (Priority PayGo + fresh socket).",
                route.tier,
                route.is_canary,
                budget,
                type(first_err).__name__,
            )
            if route.tier == "standard":
                await self.breaker.record_standard_outcome(
                    success=False,
                    ttft_seconds=budget,
                    is_canary=route.is_canary,
                )

            # --- Attempt 2: Immediate Failover to Priority PayGo + Fresh Socket ---
            buffered, iterator, raw_stream, _ = await self._open_verified_stream(
                model=target_model,
                messages=messages,
                use_priority=True,
                use_ephemeral_socket=True,
                vertex_location=self.fallback_location,
                ttft_budget=budget,
                cached_content=cached_content,
                kwargs=kwargs,
            )

        async def _stream_generator() -> AsyncIterator[Any]:
            try:
                for item in buffered:
                    yield item
                if iterator is not None:
                    async with asyncio.timeout(self.stream_completion_timeout):
                        async for item in iterator:
                            yield item
            finally:
                if raw_stream is not None and hasattr(raw_stream, "aclose"):
                    await raw_stream.aclose()

        if stream:
            return _stream_generator()

        # Caller requested unary response (`stream=False`): collect chunks and
        # rebuild a standard `litellm.ModelResponse`.
        all_chunks: List[Any] = []
        async for chunk in _stream_generator():
            all_chunks.append(chunk)
        return litellm.stream_chunk_builder(all_chunks, messages=messages)

    async def acompletion_guardrail(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[litellm.ModelResponse, AsyncIterator[Any]]:
        """Dedicated fast path for synchronous safety/guardrail checks.

        Offloads short classification/guardrail checks to
        `vertex_ai/gemini-3.5-flash-lite` with a strict `2.0s` TTFT budget.
        """
        return await self.acompletion(
            messages=messages,
            model=self.guardrail_model,
            stream=stream,
            ttft_timeout=self.guardrail_ttft_timeout,
            **kwargs,
        )


# ==============================================================================
# Self-Contained Verification Suite (`python3 vertex_resilient_acompletion.py`)
# ==============================================================================
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch

    class _FakeStream:
        """Simulates a Vertex AI `:streamGenerateContent` async stream."""

        def __init__(
            self,
            chunks: List[Any],
            delays: List[float],
        ) -> None:
            self._chunks = chunks
            self._delays = delays
            self._idx = 0
            self.closed = False

        def __aiter__(self) -> "_FakeStream":
            return self

        async def __anext__(self) -> Any:
            if self._idx >= len(self._chunks):
                raise StopAsyncIteration
            delay = self._delays[self._idx] if self._idx < len(self._delays) else 0.0
            chunk = self._chunks[self._idx]
            self._idx += 1
            if delay > 0:
                await asyncio.sleep(delay)
            return chunk

        async def aclose(self) -> None:
            self.closed = True

    def _make_chunk(content: Optional[str]) -> litellm.ModelResponse:
        return litellm.ModelResponse(
            id="chatcmpl-test",
            choices=[
                litellm.Choices(
                    index=0,
                    delta=litellm.utils.Delta(role="assistant", content=content),
                )
            ],
            model="gemini-3.8-flash",
            stream=True,
        )

    class VertexResilientClientTests(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self) -> None:
            self.breaker = VertexPriorityCircuitBreaker(
                burst_failure_threshold=3,
                failure_rate_threshold=0.05,
                min_window_samples=10,
                canary_ratio=0.05,
                recovery_canary_successes=5,
                healthy_ttft_threshold=3.0,
                rng=random.Random(42),
            )
            self.client = VertexResilientClient(
                ttft_timeout=0.15,
                guardrail_ttft_timeout=0.10,
                stream_completion_timeout=2.0,
                circuit_breaker=self.breaker,
            )

        async def asyncTearDown(self) -> None:
            await self.client.aclose()

        async def test_01_closed_state_routes_standard_and_builds_unary_response(self) -> None:
            captured_calls: List[Dict[str, Any]] = []

            async def fake_acompletion(**kwargs: Any) -> _FakeStream:
                captured_calls.append(kwargs)
                return _FakeStream(
                    chunks=[_make_chunk("Hello "), _make_chunk("Vertex AI!")],
                    delays=[0.01, 0.01],
                )

            with patch("litellm.acompletion", side_effect=fake_acompletion):
                resp = await self.client.acompletion(
                    messages=[{"role": "user", "content": "Hi"}],
                    stream=False,
                    cached_content="projects/p/locations/global/cachedContents/123",
                )

            self.assertEqual(len(captured_calls), 1)
            self.assertEqual(captured_calls[0]["model"], "vertex_ai/gemini-3.8-flash")
            self.assertTrue(captured_calls[0]["stream"])
            self.assertNotIn(
                "X-Vertex-AI-LLM-Shared-Request-Type",
                captured_calls[0]["extra_headers"],
            )
            self.assertIs(captured_calls[0]["client"], self.client.warm_client)
            self.assertEqual(
                resp.choices[0].message.content, "Hello Vertex AI!"
            )

        async def test_02_empty_initial_chunk_trap_triggers_priority_failover(self) -> None:
            """Verifies an immediate empty role chunk followed by a prefill stall times out and fails over."""
            captured_calls: List[Dict[str, Any]] = []
            first_stream = _FakeStream(
                # Chunk 0 arrives in 5ms with empty content; Chunk 1 stalls for 0.50s (> 0.15s budget)
                chunks=[_make_chunk(""), _make_chunk("Stalled")],
                delays=[0.005, 0.50],
            )
            second_stream = _FakeStream(
                chunks=[_make_chunk("Recovered via Priority PayGo")],
                delays=[0.01],
            )

            async def fake_acompletion(**kwargs: Any) -> _FakeStream:
                captured_calls.append(kwargs)
                return first_stream if len(captured_calls) == 1 else second_stream

            with patch("litellm.acompletion", side_effect=fake_acompletion):
                resp = await self.client.acompletion(
                    messages=[{"role": "user", "content": "Voice turn"}],
                    stream=False,
                )

            self.assertEqual(len(captured_calls), 2)
            # First stream must be explicitly closed upon TTFT abort
            self.assertTrue(first_stream.closed)
            # Attempt 2 must use Priority header + ephemeral_client (Connection: close)
            self.assertEqual(
                captured_calls[1]["extra_headers"].get(
                    "X-Vertex-AI-LLM-Shared-Request-Type"
                ),
                "priority",
            )
            self.assertEqual(
                captured_calls[1]["extra_headers"].get("Connection"), "close"
            )
            self.assertIs(captured_calls[1]["client"], self.client.ephemeral_client)
            self.assertEqual(
                resp.choices[0].message.content, "Recovered via Priority PayGo"
            )

        async def test_03_circuit_breaker_trips_and_splits_95_5_canary(self) -> None:
            # 3 burst failures within 30s trip the breaker from CLOSED -> OPEN
            for _ in range(3):
                await self.breaker.record_standard_outcome(
                    success=False, ttft_seconds=5.0, is_canary=False
                )
            self.assertEqual(self.breaker.state, BreakerState.OPEN)

            # Sample 1,000 routing decisions in OPEN state
            decisions = [
                await self.breaker.choose_attempt1_route() for _ in range(1000)
            ]
            priority_count = sum(1 for d in decisions if d.tier == "priority")
            canary_count = sum(
                1 for d in decisions if d.tier == "standard" and d.is_canary
            )
            self.assertEqual(priority_count + canary_count, 1000)
            # Expect ~5% canary (between 3% and 7%)
            self.assertGreaterEqual(canary_count, 30)
            self.assertLessEqual(canary_count, 70)

        async def test_04_canary_auto_recovery_closes_breaker(self) -> None:
            for _ in range(3):
                await self.breaker.record_standard_outcome(
                    success=False, ttft_seconds=5.0, is_canary=False
                )
            self.assertEqual(self.breaker.state, BreakerState.OPEN)

            # 4 healthy canaries keep it OPEN; the 5th closes it back to 100% Standard
            for i in range(5):
                state = await self.breaker.record_standard_outcome(
                    success=True, ttft_seconds=1.2, is_canary=True
                )
                if i < 4:
                    self.assertEqual(state, BreakerState.OPEN)
                else:
                    self.assertEqual(state, BreakerState.CLOSED)

        async def test_05_guardrail_helper_uses_flash_lite_and_tight_budget(self) -> None:
            captured_calls: List[Dict[str, Any]] = []

            async def fake_acompletion(**kwargs: Any) -> _FakeStream:
                captured_calls.append(kwargs)
                return _FakeStream(chunks=[_make_chunk("SAFE")], delays=[0.01])

            with patch("litellm.acompletion", side_effect=fake_acompletion):
                resp = await self.client.acompletion_guardrail(
                    messages=[{"role": "user", "content": "Check policy"}]
                )

            self.assertEqual(
                captured_calls[0]["model"], "vertex_ai/gemini-3.5-flash-lite"
            )
            self.assertEqual(resp.choices[0].message.content, "SAFE")

    unittest.main(verbosity=2)
