"""Bulk generated-answer audit: deterministic scoring and simulated inference."""
from __future__ import annotations

from iq_local.mamba3_bulk_audit import grade_response, summarize


def row(case, mode, text, target, *, group="fresh", variant="original",
        pair_id="project", error=None):
    graded = grade_response(text, [target] if target is not None else None)
    return {
        "case_id": case, "mode": mode, "group": group,
        "variant": variant, "pair_id": pair_id,
        "response": text, "error": error, "seconds": 1.25, **graded,
    }


def test_grade_requires_actual_first_answer_not_just_a_logit_or_mention():
    result = grade_response(" amber. Later more text", ["amber"])
    assert result["correct"] is True
    assert result["first_word"] == "amber"
    padded = grade_response("The assigned code is amber.", ["amber"])
    assert padded["correct"] is False
    assert padded["expected_mentioned"] is True
    assert grade_response("ambergris", ["amber"])["expected_mentioned"] is False
    assert grade_response("   ", ["amber"])["nonempty"] is False
    assert grade_response("Once upon a time", None)["correct"] is None
    assert grade_response("  7.", ["7", "seven"])["correct"] is True


def test_paired_source_switch_and_head_to_head():
    rows = [
        row("fresh:key:original", "baseline", " amber.", "amber"),
        row("fresh:key:original", "hybrid", " amber.", "amber"),
        row("fresh:key:swapped", "baseline", " amber.", "blue", variant="swapped"),
        row("fresh:key:swapped", "hybrid", " blue.", "blue", variant="swapped"),
    ]
    s = summarize(rows)
    assert s["group_metrics"]["fresh"]["hybrid"]["correct_first_answer_word"] == 2
    assert s["group_metrics"]["fresh"]["baseline"]["correct_first_answer_word"] == 1
    assert s["pair_metrics"]["fresh/hybrid"]["both_correct"] == 1
    assert s["pair_metrics"]["fresh/hybrid"]["output_changed"] == 1
    assert s["pair_metrics"]["fresh/baseline"]["both_correct"] == 0
    assert s["pair_metrics"]["fresh/baseline"]["identical_first_word"] == 1
    assert s["head_to_head"]["hybrid_wins"] == 1
    assert s["head_to_head"]["both_correct"] == 1


def test_regression_error_and_freeform_are_not_hidden():
    rows = [
        row("general:capital", "baseline", " Paris.", "Paris",
            group="general", variant="single", pair_id=None),
        row("general:capital", "hybrid", " London.", "Paris",
            group="general", variant="single", pair_id=None),
        row("general:story", "baseline", " dark woods", None,
            group="general", variant="single", pair_id=None),
        row("general:story", "hybrid", " the road", None,
            group="general", variant="single", pair_id=None),
        row("fresh:crash:original", "baseline", "", "amber",
            error="RuntimeError: out of memory"),
        row("fresh:crash:original", "hybrid", "", "amber",
            error="RuntimeError: out of memory"),
    ]
    s = summarize(rows)
    assert s["head_to_head"]["baseline_wins"] == 1
    assert s["head_to_head"]["runtime_errors"] == 1
    assert s["group_metrics"]["general"]["hybrid"]["graded"] == 1
    assert s["group_metrics"]["general"]["hybrid"]["generations"] == 2
    assert s["finding_counts"]["hybrid_accuracy_regression"] == 1
    assert s["finding_counts"]["runtime_error"] == 1
    assert s["promotion_status"] == "research_only_not_certified"


def test_missing_modes_are_flagged():
    s = summarize([row("fresh:key:original", "hybrid", "amber", "amber")])
    assert s["finding_counts"]["missing_mode"] == 1
