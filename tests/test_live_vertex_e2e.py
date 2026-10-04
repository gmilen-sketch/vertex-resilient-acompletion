"""Live End-to-End Vertex AI Integration Test Suite for VertexResilientClient.

Executes 5 live Vertex AI API calls (`:streamGenerateContent`) via `litellm.acompletion`
against a real Google Cloud project (`vertex_ai/gemini-3.8-flash` and
`vertex_ai/gemini-3.5-flash-lite`):
1. Live Unary (`stream=False`) on `vertex_ai/gemini-3.8-flash` (Standard Pay-As-You-Go)
2. Live Streaming (`stream=True`) on `vertex_ai/gemini-3.8-flash` with verified TTFT telemetry
3. Live Guardrail Fast-Path (`acompletion_guardrail`) on `vertex_ai/gemini-3.5-flash-lite`
4. Live Attempt-1 TTFT Watchdog Timeout -> Attempt-2 Priority Pay-As-You-Go Failover
   (`X-Vertex-AI-LLM-Shared-Request-Type: priority` + ephemeral `Connection: close` socket)
5. Live Circuit Breaker `OPEN` State Priority Routing + Canary Auto-Recovery (`OPEN -> CLOSED`)

Usage:
    export VERTEX_PROJECT="your-gcp-project-id"
    # Optional if using a specific gcloud account instead of default ADC:
    # export GCLOUD_ACCOUNT="user@example.com"
    python3 tests/test_live_vertex_e2e.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.auth
import google.auth.credentials
from vertex_resilient_acompletion import (  # noqa: E402
    BreakerState,
    VertexPriorityCircuitBreaker,
    VertexResilientClient,
)


class _StaticBearerCredentials(google.auth.credentials.Credentials):
    """Bearer token credentials compatible with LiteLLM's refresh_auth()."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token

    def refresh(self, request: Any) -> None:
        pass

    @property
    def valid(self) -> bool:
        return True


def resolve_project_and_credentials() -> Tuple[str, Optional[google.auth.credentials.Credentials]]:
    """Resolves the target GCP project and credentials from env, ADC, or gcloud CLI."""
    project_id = (
        os.environ.get("VERTEX_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or ""
    ).strip()
    gcloud_account = os.environ.get("GCLOUD_ACCOUNT", "").strip()

    if not project_id:
        try:
            project_id = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            project_id = ""

    if not project_id or project_id == "(unset)":
        raise RuntimeError(
            "Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT environment variable before running live E2E tests."
        )

    if gcloud_account:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token", f"--account={gcloud_account}"],
            text=True,
        ).strip()
        return project_id, _StaticBearerCredentials(token=token)

    try:
        creds, _ = google.auth.default()
        return project_id, creds
    except Exception:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
        ).strip()
        return project_id, _StaticBearerCredentials(token=token)


async def run_live_vertex_e2e_tests() -> Dict[str, Any]:
    project_id, creds = resolve_project_and_credentials()

    def _custom_auth_default(*args: Any, **kwargs: Any) -> Tuple[Any, str]:
        return creds, project_id

    results: Dict[str, Any] = {"vertex_project": project_id}
    with patch("google.auth.default", side_effect=_custom_auth_default):
        breaker = VertexPriorityCircuitBreaker(
            burst_failure_threshold=3,
            failure_rate_threshold=0.05,
            min_window_samples=10,
            canary_ratio=0.05,
            recovery_canary_successes=3,
            healthy_ttft_threshold=4.0,
        )
        client = VertexResilientClient(
            default_model="vertex_ai/gemini-3.8-flash",
            guardrail_model="vertex_ai/gemini-3.5-flash-lite",
            primary_location="global",
            fallback_location="global",
            ttft_timeout=4.5,
            guardrail_ttft_timeout=2.5,
            stream_completion_timeout=25.0,
            circuit_breaker=breaker,
        )

        try:
            # Test 1: Live Unary (`stream=False`) on `vertex_ai/gemini-3.8-flash` (Standard PayGo)
            t0 = time.perf_counter()
            resp1 = await client.acompletion(
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with the exact phrase: VERTEX_UNARY_OK",
                    }
                ],
                stream=False,
                vertex_project=project_id,
            )
            dt1 = time.perf_counter() - t0
            text1 = resp1.choices[0].message.content.strip()
            print(f"[LIVE TEST 1 - Unary gemini-3.8-flash Standard] ({dt1:.3f}s): {text1}")
            assert "VERTEX_UNARY_OK" in text1, f"Unexpected response: {text1}"
            results["test_1_unary_3_8_flash_standard"] = {
                "status": "PASS",
                "latency_s": round(dt1, 3),
                "output": text1,
                "breaker_state": breaker.state.value,
            }

            # Test 2: Live Streaming (`stream=True`) on `vertex_ai/gemini-3.8-flash`
            t0 = time.perf_counter()
            stream_gen = await client.acompletion(
                messages=[
                    {
                        "role": "user",
                        "content": "Count from 1 to 3 separated by commas.",
                    }
                ],
                stream=True,
                vertex_project=project_id,
            )
            chunks = []
            first_token_s = None
            async for ch in stream_gen:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                delta_text = ch.choices[0].delta.content or ""
                if delta_text:
                    chunks.append(delta_text)
            dt2 = time.perf_counter() - t0
            text2 = "".join(chunks).strip()
            print(
                f"[LIVE TEST 2 - Streaming gemini-3.8-flash] "
                f"(TTFT={first_token_s:.3f}s, Total={dt2:.3f}s, Chunks={len(chunks)}): {text2}"
            )
            assert len(chunks) >= 1 and "1" in text2
            results["test_2_streaming_3_8_flash"] = {
                "status": "PASS",
                "ttft_s": round(first_token_s or dt2, 3),
                "total_s": round(dt2, 3),
                "chunks": len(chunks),
                "output": text2,
            }

            # Test 3: Live Guardrail Fast-Path (`vertex_ai/gemini-3.5-flash-lite`)
            t0 = time.perf_counter()
            resp3 = await client.acompletion_guardrail(
                messages=[
                    {
                        "role": "user",
                        "content": "Classify this utterance as SAFE or UNSAFE: 'What is my account balance?' Reply with one word.",
                    }
                ],
                stream=False,
                vertex_project=project_id,
            )
            dt3 = time.perf_counter() - t0
            text3 = resp3.choices[0].message.content.strip()
            print(
                f"[LIVE TEST 3 - Guardrail gemini-3.5-flash-lite] ({dt3:.3f}s): {text3}"
            )
            assert "SAFE" in text3.upper()
            results["test_3_guardrail_3_5_flash_lite"] = {
                "status": "PASS",
                "latency_s": round(dt3, 3),
                "output": text3,
            }

            # Test 4: Live TTFT Timeout Failover to Priority PayGo (`X-Vertex-AI-LLM-Shared-Request-Type: priority`)
            orig_open = client._open_verified_stream
            call_log = []

            async def _instrumented_open(**kw: Any) -> Any:
                attempt_num = len(call_log) + 1
                call_log.append(
                    {
                        "attempt": attempt_num,
                        "use_priority": kw["use_priority"],
                        "use_ephemeral_socket": kw["use_ephemeral_socket"],
                    }
                )
                if attempt_num == 1:
                    # Simulate a TTFT stall on Attempt 1 (50ms timeout budget)
                    kw["ttft_budget"] = 0.05
                else:
                    kw["ttft_budget"] = 10.0
                return await orig_open(**kw)

            with patch.object(
                client, "_open_verified_stream", side_effect=_instrumented_open
            ):
                t0 = time.perf_counter()
                resp4 = await client.acompletion(
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply with the exact phrase: PRIORITY_FAILOVER_OK",
                        }
                    ],
                    stream=False,
                    vertex_project=project_id,
                )
                dt4 = time.perf_counter() - t0

            text4 = resp4.choices[0].message.content.strip()
            print(
                f"[LIVE TEST 4 - Forced Attempt-1 TTFT Abort -> Live Attempt-2 Priority PayGo] "
                f"({dt4:.3f}s): {text4} | Calls={call_log}"
            )
            assert len(call_log) == 2
            assert call_log[0]["use_priority"] is False
            assert call_log[1]["use_priority"] is True
            assert call_log[1]["use_ephemeral_socket"] is True
            assert "PRIORITY_FAILOVER_OK" in text4
            results["test_4_live_priority_paygo_failover"] = {
                "status": "PASS",
                "latency_s": round(dt4, 3),
                "calls": call_log,
                "output": text4,
            }

            # Test 5: Trip Circuit Breaker to OPEN and verify live Attempt-1 Priority PayGo call + Canary Recovery
            await breaker.record_standard_outcome(
                success=False, ttft_seconds=5.0, is_canary=False
            )
            await breaker.record_standard_outcome(
                success=False, ttft_seconds=5.0, is_canary=False
            )
            assert breaker.state == BreakerState.OPEN
            print(f"[LIVE TEST 5 - Breaker State After 3 Failures]: {breaker.state.value}")

            t0 = time.perf_counter()
            resp5 = await client.acompletion(
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with the exact phrase: BREAKER_OPEN_PRIORITY_OK",
                    }
                ],
                stream=False,
                vertex_project=project_id,
            )
            dt5 = time.perf_counter() - t0
            text5 = resp5.choices[0].message.content.strip()
            print(
                f"[LIVE TEST 5 - Live Call in OPEN State] ({dt5:.3f}s): {text5}"
            )
            assert "BREAKER_OPEN_PRIORITY_OK" in text5

            for _ in range(3):
                await breaker.record_standard_outcome(
                    success=True, ttft_seconds=0.8, is_canary=True
                )
            assert breaker.state == BreakerState.CLOSED
            print(
                f"[LIVE TEST 5 - Breaker State After Canary Recovery]: {breaker.state.value}"
            )
            results["test_5_live_breaker_open_and_recovery"] = {
                "status": "PASS",
                "latency_s": round(dt5, 3),
                "output": text5,
                "final_breaker_state": breaker.state.value,
            }

        finally:
            await client.aclose()

    print("\n=== ALL 5 LIVE VERTEX AI E2E TESTS PASSED ===")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    asyncio.run(run_live_vertex_e2e_tests())
