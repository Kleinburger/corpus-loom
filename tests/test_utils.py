import time

import pytest

from corpusloom.utils import RateLimiter, approx_tokens, cosine, extract_json_str


def test_extract_json_with_fence_and_valid():
    text = '```json\n{"a": 1, "b": [2,3]}\n```'
    assert (
        extract_json_str(text) == '{"a": 1, "b": [2, 3]}'
        or extract_json_str(text) == '{"a": 1, "b": [2,3]}'
    )


def test_extract_json_no_fence_valid_array():
    text = "Here: [1, 2, 3] trailing"
    assert extract_json_str(text) == "[1, 2, 3]" or extract_json_str(text) == "[1,2,3]"


def test_extract_json_mismatch_braces_returns_none():
    # exercises branch: mismatched closing bracket
    assert extract_json_str("{]") is None


def test_extract_json_no_start_brace_returns_none():
    # no JSON-looking content
    assert extract_json_str("no json here") is None


def test_extract_json_candidate_invalid_returns_none():
    # invalid json between braces
    assert extract_json_str("{ invalid }") is None


def test_cosine_edge_cases():
    assert cosine([], []) == -1.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == -1.0
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_approx_tokens_bounds():
    assert approx_tokens("") == 1
    assert approx_tokens("abcd") == 1  # short strings still at least 1
    assert approx_tokens("a" * 9) >= 2  # heuristic scales with length


def test_rate_limiter_disabled_is_noop():
    rl = RateLimiter(0)  # 0 = disabled
    rl.throttle()  # should not raise or sleep


def test_extract_json_invalid_candidate_then_extra_closing_returns_none():
    # First balanced substring "[,]" is invalid JSON -> loads() fails,
    # then we hit a stray closing ']' with an empty stack -> returns None.
    assert extract_json_str("[,]]") is None


def test_rate_limiter_consumes_tokens_without_sleep(monkeypatch):
    # Cap > 0, tokens start full. For first three calls, no sleeping occurs; tokens are just decremented.
    rl = RateLimiter(3)

    slept = {"called": 0}
    monkeypatch.setattr(time, "sleep", lambda s: slept.__setitem__("called", slept["called"] + 1))

    # 3 tokens -> after three throttles we should not have slept yet (sleep happens only when tokens < 1)
    rl.throttle()
    rl.throttle()
    rl.throttle()

    assert slept["called"] == 0  # exercised the "else: self.tokens -= 1.0" branch


def test_rate_limiter_sleeps_when_depleted(monkeypatch):
    # Deplete tokens then call again to trigger the sleep branch (tokens < 1.0)
    rl = RateLimiter(2)

    slept = {"count": 0, "secs": []}

    def fake_sleep(s):
        slept["count"] += 1
        slept["secs"].append(s)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    # Deplete: 2 -> 1 -> 0 (no sleep yet)
    rl.throttle()
    rl.throttle()

    # Now tokens < 1, so this call should enter the sleep branch
    rl.throttle()

    assert slept["count"] == 1
    assert slept["secs"][0] > 0.0
    # After sleeping, code sets tokens to 0.0 and updates last_refill; we don't assert exact times here.


def test_rate_limiter_refill_branch(monkeypatch):
    # Freeze time to a known point for deterministic math
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "time", lambda: now["t"])

    # Prevent real sleeping; record if it would have slept
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    rl = RateLimiter(2)  # cap=2 tokens/min; tokens=2.0; last_refill=1000.0

    # Simulate partial depletion and elapsed time so refill is strictly > 0
    rl.tokens = 0.5
    now["t"] = 1060.0  # 60s later -> refill = (60) * (2/60) = 2.0

    rl.throttle()

    # Refill branch executes: tokens = min(2.0, 0.5 + 2.0) = 2.0, last_refill := now (1060.0)
    # Then tokens >= 1 so the 'else' path decrements once -> tokens == 1.0
    assert rl.tokens == pytest.approx(1.0)
    assert rl.last_refill == pytest.approx(1060.0)
    # No sleep on this path
    assert slept == []


def test_rate_limiter_refill_then_sleep_branch(monkeypatch):
    """
    Hit both:
      - refill > 0
      - tokens < 1.0
    in a single throttle() call.
    """
    # Freeze time
    now = {"t": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: now["t"])

    # Capture sleep calls
    slept = {"n": 0, "secs": []}

    def fake_sleep(s):
        slept["n"] += 1
        slept["secs"].append(s)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    rl = RateLimiter(2)  # cap=2 tokens/min => 2/60 tokens per sec
    rl.tokens = 0.4  # below 1.0 so sleep path is possible
    rl.last_refill = now["t"] - 5.0  # 5s elapsed => refill = 5 * (2/60) ≈ 0.1667

    # This call will:
    #   1) take the refill branch (tokens ≈ 0.5667)
    #   2) still be < 1.0, so it enters the sleep branch
    rl.throttle()

    assert slept["n"] == 1
    assert slept["secs"][0] >= 0.0
    # After sleep path, code sets tokens = 0.0 and updates last_refill to time()
    assert rl.tokens == pytest.approx(0.0)
    assert rl.last_refill == pytest.approx(now["t"])


def test_rate_limiter_no_refill_branch(monkeypatch):
    # Freeze time so now == last_refill -> refill == 0
    fixed = 1234.0
    monkeypatch.setattr(time, "time", lambda: fixed)

    slept = {"n": 0}
    monkeypatch.setattr(time, "sleep", lambda s: slept.__setitem__("n", slept["n"] + 1))

    rl = RateLimiter(2)  # cap=2, tokens=2.0, last_refill=fixed
    rl.throttle()  # refill == 0, tokens >= 1 → decrement path, no sleep

    assert rl.tokens == pytest.approx(1.0)
    assert rl.last_refill == pytest.approx(fixed)  # unchanged when no refill
    assert slept["n"] == 0
