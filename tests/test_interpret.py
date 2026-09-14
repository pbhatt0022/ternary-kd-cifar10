"""The section 8 interpretation rules. These decide what the study concludes, so each
branch is pinned to a case with hand-checked numbers."""

import importlib.util
import pathlib

SPEC = importlib.util.spec_from_file_location(
    "aggregate", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "aggregate.py"
)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def summaries(a, b, c, d, t=None, spreads=(0.2, 0.2, 0.2, 0.2)):
    """Minimal summary dicts: only the fields interpret() reads."""
    out = {}
    for key, mean, sp in zip("ABCD", (a, b, c, d), spreads):
        out[key] = {"last10_test_mean": mean, "last10_test_spread": sp, "arm": key}
    if t is not None:
        out["T"] = {"last10_test_mean": t, "last10_test_spread": None, "arm": "T"}
    return out


def test_spread_is_max_minus_min():
    assert aggregate.spread([93.1, 93.4, 92.9]) == 93.4 - 92.9
    assert aggregate.spread([93.1]) is None


def test_recovery_undefined_when_c_meets_or_beats_a():
    v = aggregate.interpret(summaries(a=93.0, b=93.5, c=93.2, d=93.6, t=95.0))
    assert v["recovery_ratio"] is None
    assert "no 'recovery' language" in v["recovery_verdict"]
    assert "matched or exceeded" in v["recovery_verdict"]


def test_recovery_undefined_when_gap_is_inside_the_spread():
    # |A - C| = 0.10pp, smaller than the largest within-arm spread of 0.40pp
    v = aggregate.interpret(
        summaries(a=93.0, b=93.4, c=92.9, d=93.3, t=95.0, spreads=(0.4, 0.2, 0.2, 0.2))
    )
    assert v["recovery_ratio"] is None
    assert "smaller than the largest within-arm spread" in v["recovery_verdict"]


def test_recovery_defined_and_correct():
    # A=93, C=90, D=91.5 -> R = (91.5-90)/(93-90) = 0.5
    v = aggregate.interpret(summaries(a=93.0, b=93.4, c=90.0, d=91.5, t=95.0))
    assert v["recovery_verdict"] == "defined"
    assert abs(v["recovery_ratio"] - 0.5) < 1e-9


def test_weak_teacher_blocks_the_interaction_term():
    """T at or below A's mean makes B and D uninterpretable as distillation."""
    v = aggregate.interpret(summaries(a=93.0, b=93.4, c=90.0, d=91.5, t=92.0))
    assert v["interaction"] is None
    assert v["interaction_verdict"] == "not computed; see the teacher comparison"
    assert "uninterpretable" in v["teacher_verdict"]


def test_teacher_exactly_equal_to_a_still_blocks():
    v = aggregate.interpret(summaries(a=93.0, b=93.4, c=90.0, d=91.5, t=93.0))
    assert v["interaction"] is None


def test_interaction_suggestive_only_beyond_twice_the_max_spread():
    # (D-C) - (B-A) = (92.0-90.0) - (93.4-93.0) = 1.6pp; 2 x 0.2 = 0.4 -> suggestive
    v = aggregate.interpret(summaries(a=93.0, b=93.4, c=90.0, d=92.0, t=95.0))
    assert abs(v["interaction"] - 1.6) < 1e-9
    assert v["interaction_verdict"].startswith("suggestive")


def test_interaction_unresolved_inside_the_threshold():
    # (D-C) - (B-A) = (90.5-90.0) - (93.4-93.0) = 0.1pp; 2 x 0.6 = 1.2 -> unresolved
    v = aggregate.interpret(
        summaries(a=93.0, b=93.4, c=90.0, d=90.5, t=95.0, spreads=(0.6, 0.3, 0.3, 0.3))
    )
    assert v["interaction_verdict"] == "unresolved"


def test_max_spread_includes_arm_b():
    """B's spread must not be excluded just because it comes from 2 seeds."""
    v = aggregate.interpret(
        summaries(a=93.0, b=93.4, c=90.0, d=91.5, t=95.0, spreads=(0.2, 0.9, 0.2, 0.2))
    )
    assert v["max_within_arm_spread"] == 0.9
    assert "2 seeds" in v["max_spread_note"]


def test_incomplete_arms_report_incomplete():
    v = aggregate.interpret({"A": {"last10_test_mean": 93.0, "last10_test_spread": 0.2,
                                   "arm": "A"}})
    assert "incomplete" in v["status"]
