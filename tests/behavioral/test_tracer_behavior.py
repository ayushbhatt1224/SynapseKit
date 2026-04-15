"""Behavioral tests — TokenTracer.

Verifies:
  - Token counting accumulates correctly across calls
  - Disabled tracer records nothing
  - Summary output is well-formed with all required keys
  - Cost estimate is present and non-negative
"""
from __future__ import annotations

from synapsekit.observability.tracer import TokenTracer


class TestTokenTracerBasics:
    def test_starts_at_zero(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        s = t.summary()
        assert s["calls"] == 0
        assert s["total_tokens"] == 0
        assert s["total_input_tokens"] == 0
        assert s["total_output_tokens"] == 0

    def test_record_call_increments_calls(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=10, output_tokens=5, latency_ms=100.0)
        assert t.summary()["calls"] == 1

    def test_record_accumulates_tokens(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        t.record(input_tokens=20, output_tokens=8, latency_ms=80.0)
        s = t.summary()
        assert s["total_input_tokens"] == 30
        assert s["total_output_tokens"] == 13
        assert s["total_tokens"] == 43

    def test_multiple_records(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        for _ in range(10):
            t.record(input_tokens=5, output_tokens=3, latency_ms=10.0)
        s = t.summary()
        assert s["calls"] == 10
        assert s["total_input_tokens"] == 50
        assert s["total_output_tokens"] == 30
        assert s["total_tokens"] == 80

    def test_latency_accumulates(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=5, output_tokens=3, latency_ms=100.0)
        t.record(input_tokens=5, output_tokens=3, latency_ms=200.0)
        s = t.summary()
        assert s["total_latency_ms"] == 300.0


class TestTokenTracerDisabled:
    def test_disabled_records_nothing(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=False)
        t.record(input_tokens=100, output_tokens=50, latency_ms=200.0)
        s = t.summary()
        assert s["calls"] == 0
        assert s["total_tokens"] == 0

    def test_disabled_multiple_records(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=False)
        for _ in range(5):
            t.record(input_tokens=10, output_tokens=5, latency_ms=10.0)
        assert t.summary()["calls"] == 0


class TestTokenTracerSummaryFormat:
    def test_summary_has_required_keys(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        s = t.summary()
        required = {
            "calls",
            "total_tokens",
            "total_input_tokens",
            "total_output_tokens",
            "total_latency_ms",
            "estimated_cost_usd",
        }
        assert required.issubset(s.keys())

    def test_summary_values_are_non_negative(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=5, output_tokens=3, latency_ms=10.0)
        s = t.summary()
        for key, val in s.items():
            if isinstance(val, (int, float)):
                assert val >= 0, f"{key} should be non-negative"

    def test_total_equals_input_plus_output(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=100, output_tokens=200, latency_ms=10.0)
        t.record(input_tokens=50, output_tokens=75, latency_ms=10.0)
        s = t.summary()
        assert s["total_tokens"] == s["total_input_tokens"] + s["total_output_tokens"]

    def test_model_in_summary(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        assert t.summary()["model"] == "gpt-4o-mini"


class TestTokenTracerCostEstimate:
    def test_cost_estimate_non_negative(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=1000, output_tokens=500, latency_ms=100.0)
        s = t.summary()
        assert isinstance(s["estimated_cost_usd"], float)
        assert s["estimated_cost_usd"] >= 0

    def test_cost_zero_for_unknown_model(self):
        """Unknown model has no cost table entry — estimated cost is 0."""
        t = TokenTracer(model="unknown-model-xyz", enabled=True)
        t.record(input_tokens=1000, output_tokens=500, latency_ms=100.0)
        assert t.summary()["estimated_cost_usd"] == 0.0

    def test_reset_clears_all_records(self):
        t = TokenTracer(model="gpt-4o-mini", enabled=True)
        t.record(input_tokens=100, output_tokens=50, latency_ms=50.0)
        t.reset()
        s = t.summary()
        assert s["calls"] == 0
        assert s["total_tokens"] == 0
