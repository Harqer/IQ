"""Canary checks reject garbled output without claiming broad model quality."""

import pytest

from iq_local.quality import check_arithmetic_canary


@pytest.mark.parametrize(
    "output",
    ["4", "4.", "4.0", "four", "The answer is 4.", "Answer: 4"],
)
def test_arithmetic_canary_accepts_exact_answer(output):
    assert check_arithmetic_canary(output)[0]


@pytest.mark.parametrize(
    "output",
    ["", "3", "The answer is 8.", "4, hallucinated text", "/sites t OSIiensiensiens milit doctr",
     "Hadateraterateraterater", "fourteen", "4 5", "4" * 100],
)
def test_arithmetic_canary_rejects_incorrect_or_garbled_output(output):
    passed, reason = check_arithmetic_canary(output)
    assert not passed
    assert reason


def test_arithmetic_canary_is_not_a_quality_metric():
    # Even a correct 2+2 reply proves only the one canary.
    assert check_arithmetic_canary("4")[1] == "2+2 canary passed"
