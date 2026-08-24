from __future__ import annotations

from app.inference.postprocess import age_to_bracket


def test_age_bracket_boundaries_are_unambiguous() -> None:
    assert age_to_bracket(17.99) == "unknown"
    assert age_to_bracket(18.0) == "18-30"
    assert age_to_bracket(30.99) == "18-30"
    assert age_to_bracket(31.0) == "31-45"
    assert age_to_bracket(46.0) == "46-60"
    assert age_to_bracket(59.99) == "46-60"
    assert age_to_bracket(60.0) == "60+"
    assert age_to_bracket(120.0) == "60+"
    assert age_to_bracket(120.01) == "unknown"
