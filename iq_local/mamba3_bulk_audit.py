"""Pure-text grading and bulk diagnostics for real Mamba-3 generated responses.

No training or checkpoint mutation. Grade the first generated answer word for
single-word fact questions; also preserve every complete generated response.
"""
from __future__ import annotations

import re
from collections import defaultdict


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def grade_response(response: str, answers: list[str] | None) -> dict:
    words = _WORD.findall(response)
    first = words[0].casefold() if words else None
    valid = {word.casefold() for word in answers} if answers else None
    return {
        "first_word": first,
        "nonempty": bool(response.strip()),
        "correct": first in valid if valid is not None else None,
        "expected_mentioned": any(word.casefold() in valid for word in words)
        if valid is not None else None,
    }


def summarize(rows: list[dict]) -> dict:
    """Count completed generations, paired evidence flips, and failure types."""
    by_group = defaultdict(lambda: defaultdict(list))
    by_case = defaultdict(dict)
    pairs = defaultdict(dict)
    for row in rows:
        by_group[row["group"]][row["mode"]].append(row)
        by_case[row["case_id"]][row["mode"]] = row
        if row.get("pair_id"):
            pairs[(row["group"], row["pair_id"], row["mode"])][row["variant"]] = row

    groups = {}
    for group, modes in sorted(by_group.items()):
        groups[group] = {}
        for mode, items in sorted(modes.items()):
            graded = [row for row in items if row.get("correct") is not None
                      and row.get("error") is None]
            groups[group][mode] = {
                "generations": len(items),
                "runtime_errors": sum(bool(row.get("error")) for row in items),
                "nonempty_responses": sum(bool(row.get("nonempty")) for row in items),
                "graded": len(graded),
                "correct_first_answer_word": sum(bool(row["correct"]) for row in graded),
                "expected_mentioned": sum(bool(row.get("expected_mentioned")) for row in graded),
                "total_seconds": round(sum(float(row.get("seconds", 0)) for row in items), 3),
            }

    paired = defaultdict(lambda: {
        "pairs": 0, "both_correct": 0, "output_changed": 0,
        "identical_first_word": 0, "missing_response": 0,
    })
    for (group, _key, mode), variants in sorted(pairs.items()):
        if set(variants) != {"original", "swapped"}:
            continue
        a, b = variants["original"], variants["swapped"]
        agg = paired[(group, mode)]
        agg["pairs"] += 1
        agg["both_correct"] += int(a.get("correct") is True and b.get("correct") is True)
        changed = (a.get("first_word") != b.get("first_word"))
        agg["output_changed"] += int(changed)
        agg["identical_first_word"] += int(not changed)
        agg["missing_response"] += int(not a.get("nonempty") or not b.get("nonempty"))

    failures = []
    outcome = {"hybrid_wins": 0, "baseline_wins": 0, "both_correct": 0,
               "both_wrong": 0, "runtime_errors": 0}
    for case_id, modes in sorted(by_case.items()):
        if set(modes) != {"baseline", "hybrid"}:
            failures.append({"case_id": case_id, "kind": "missing_mode"})
            continue
        base, active = modes["baseline"], modes["hybrid"]
        if base.get("error") or active.get("error"):
            outcome["runtime_errors"] += 1
            failures.append({"case_id": case_id, "kind": "runtime_error",
                             "baseline": base.get("error"), "hybrid": active.get("error")})
            continue
        if not active.get("nonempty"):
            failures.append({"case_id": case_id, "kind": "empty_hybrid_response"})
        if base.get("correct") is None:
            continue
        b = bool(base["correct"])
        a = bool(active["correct"])
        if a and not b:
            outcome["hybrid_wins"] += 1
        elif b and not a:
            outcome["baseline_wins"] += 1
            failures.append({"case_id": case_id, "kind": "hybrid_accuracy_regression",
                             "baseline_response": base.get("response"),
                             "hybrid_response": active.get("response")})
        elif a and b:
            outcome["both_correct"] += 1
        else:
            outcome["both_wrong"] += 1
            failures.append({"case_id": case_id, "kind": "both_models_wrong",
                             "baseline_response": base.get("response"),
                             "hybrid_response": active.get("response")})

    for (group, mode), agg in paired.items():
        if mode == "hybrid" and agg["both_correct"] < agg["pairs"]:
            failures.append({
                "kind": "incomplete_evidence_switching", "group": group,
                "both_correct": agg["both_correct"], "pairs": agg["pairs"],
                "same_first_word": agg["identical_first_word"],
            })

    return {
        "group_metrics": groups,
        "pair_metrics": {f"{g}/{m}": v for (g, m), v in sorted(paired.items())},
        "head_to_head": outcome,
        "findings": failures,
        "finding_counts": dict(sorted({
            kind: sum(row["kind"] == kind for row in failures)
            for kind in {row["kind"] for row in failures}
        }.items())),
        "promotion_status": "research_only_not_certified",
        "interpretation": (
            "Scores actual decoded greedy generations. First-answer-word matching "
            "is intentionally strict for one-word code questions; supplementary "
            "expected_mentioned is diagnostic, not the accuracy score. Previously "
            "inspected locked cases are regression tests, NOT clean validation."
        ),
    }
