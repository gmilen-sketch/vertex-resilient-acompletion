"""Isolated Asyncio Unit Test Suite for VertexResilientClient & VertexPriorityCircuitBreaker.

Runs all 7 deterministic transport, circuit-breaker, reasoning, and security floor tests:
1. `CLOSED` state routes 100% to Standard PayGo and reassembles unary `ModelResponse` from stream.
2. Empty initial metadata/role chunk trap (`role="assistant"`, `content=""`) followed by a
   prefill stall triggers `4.5s` TTFT abort (`await stream.aclose()`) and Attempt-2 Priority
   PayGo failover (`X-Vertex-AI-LLM-Shared-Request-Type: priority` + `Connection: close`).
3. Circuit breaker trips to `OPEN` after burst failures and splits traffic `95%` Priority /
   `5%` Standard Canary Probe.
4. Consecutive healthy Standard Canary Probes (`TTFT < 3.0s`) auto-recover the breaker to `CLOSED`.
5. `acompletion_guardrail` routes to `vertex_ai/gemini-3.5-flash-lite` with a tighter `2.0s` TTFT budget.
6. `acompletion_reasoning` routes to `vertex_ai/gemini-3.5-pro` and sanitizes Gemini 4 kwargs.
7. `requirements.txt` enforces `litellm>=1.84.0` minimum security floor (`CVE-2026-49468`).

Usage:
    python3 tests/test_unit_resilient_client.py
"""

import asyncio
import os
import random
import re
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vertex_resilient_acompletion import (  # noqa: E402
    BreakerState,
    VertexPriorityCircuitBreaker,
    VertexResilientClient,
    sanitize_gemini_4_kwargs,
)
import litellm  # noqa: E402


class _FakeStream:
    def __init__(self, chunks: List[Any], delays: List[float]) -> None:
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
        self.assertEqual(resp.choices[0].message.content, "Hello Vertex AI!")

    async def test_02_empty_initial_chunk_trap_triggers_priority_failover(self) -> None:
        """Verifies an immediate empty role chunk followed by a prefill stall times out and fails over."""
        captured_calls: List[Dict[str, Any]] = []
        first_stream = _FakeStream(
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
        self.assertTrue(first_stream.closed)
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
        for _ in range(3):
            await self.breaker.record_standard_outcome(
                success=False, ttft_seconds=5.0, is_canary=False
            )
        self.assertEqual(self.breaker.state, BreakerState.OPEN)

        decisions = [
            await self.breaker.choose_attempt1_route() for _ in range(1000)
        ]
        priority_count = sum(1 for d in decisions if d.tier == "priority")
        canary_count = sum(
            1 for d in decisions if d.tier == "standard" and d.is_canary
        )
        self.assertEqual(priority_count + canary_count, 1000)
        self.assertGreaterEqual(canary_count, 30)
        self.assertLessEqual(canary_count, 70)

    async def test_04_canary_auto_recovery_closes_breaker(self) -> None:
        for _ in range(3):
            await self.breaker.record_standard_outcome(
                success=False, ttft_seconds=5.0, is_canary=False
            )
        self.assertEqual(self.breaker.state, BreakerState.OPEN)

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

    async def test_06_reasoning_routing_and_gemini_4_parameter_sanitization(self) -> None:
        captured_calls: List[Dict[str, Any]] = []

        async def fake_acompletion(**kwargs: Any) -> _FakeStream:
            captured_calls.append(kwargs)
            return _FakeStream(chunks=[_make_chunk("REASONING_OK")], delays=[0.01])

        with patch("litellm.acompletion", side_effect=fake_acompletion):
            resp = await self.client.acompletion_reasoning(
                messages=[{"role": "user", "content": "Solve deep task"}],
                thinking_budget=8192,
                candidate_count=2,
                frequency_penalty=0.3,
            )

        self.assertEqual(captured_calls[0]["model"], "vertex_ai/gemini-3.5-pro")
        self.assertEqual(captured_calls[0]["vertex_location"], "global")
        cleaned = sanitize_gemini_4_kwargs(
            "vertex_ai/gemini-4-preview",
            {
                "thinking_budget": 8192,
                "candidate_count": 2,
                "frequency_penalty": 0.3,
                "temperature": 0.2,
            },
        )
        self.assertNotIn("thinking_budget", cleaned)
        self.assertNotIn("candidate_count", cleaned)
        self.assertNotIn("frequency_penalty", cleaned)
        self.assertEqual(cleaned["temperature"], 0.2)
        self.assertEqual(resp.choices[0].message.content, "REASONING_OK")

    def test_07_cve_2026_49468_litellm_minimum_version_floor(self) -> None:
        """Verifies requirements.txt pins litellm >= 1.84.0 (CVE-2026-49468 remediation)."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        req_path = os.path.join(repo_root, "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as fh:
            lines = [
                ln.split("#", 1)[0].strip()
                for ln in fh
                if ln.split("#", 1)[0].strip()
            ]
        litellm_specs = [ln for ln in lines if ln.lower().startswith("litellm")]
        self.assertEqual(len(litellm_specs), 1, f"Expected litellm requirement in {req_path}")
        match = re.search(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", litellm_specs[0])
        self.assertIsNotNone(
            match,
            f"litellm requirement '{litellm_specs[0]}' must specify a '>=' minimum floor",
        )
        assert match is not None
        floor: Tuple[int, int, int] = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 0),
        )
        self.assertGreaterEqual(
            floor,
            (1, 84, 0),
            f"CVE-2026-49468 requires litellm >= 1.84.0, found {litellm_specs[0]}",
        )
        self.assertIn("httpx>=0.28.1", lines)
        self.assertIn("google-cloud-aiplatform>=1.70.0", lines)
        dep_path = os.path.join(repo_root, ".github", "dependabot.yml")
        with open(dep_path, "r", encoding="utf-8") as df:
            dep_txt = df.read()
        self.assertIn("increase-if-necessary", dep_txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
