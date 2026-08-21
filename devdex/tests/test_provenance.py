"""Guards for provenance.vendor_error() — the 200-status-but-really-a-refusal detector.

Found on the docs track: the item at
docs/v3/concepts/rate-limits.mdx was scored as an infrastructure error on every arm, not because
any vendor throttled, but because the regex matched the bare phrase "rate limit" inside the
result's own URL/title. A real engine that correctly retrieves any page discussing rate limiting
would have its retrieval discarded the same way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from provenance import vendor_error  # noqa: E402


def test_a_result_about_rate_limiting_is_not_a_vendor_error():
    payload = {"data": [{"url": "https://github.com/PrefectHQ/prefect/blob/main/"
                                 "docs/v3/concepts/rate-limits.mdx",
                          "title": "PrefectHQ/prefect",
                          "markdown": "Prefect Cloud enforces a 5 MiB maximum on request size."}]}
    assert vendor_error(payload) is None


def test_a_genuine_throttle_message_is_still_caught():
    # the documented real trigger: Context7 answering a rate-limited call with HTTP 200
    payload = {"content": [{"type": "text",
                            "text": "Rate limit exceeded. Please try again in 547 seconds."}]}
    assert vendor_error(payload) is not None


def test_rate_limited_as_one_word_is_still_caught():
    assert vendor_error("You have been rate-limited, please slow down.") is not None
