from shapely.geometry import LineString, MultiLineString, Polygon

import logging
import inference.grow_cut as gc

from inference.grow_cut import (
    get_spanning_line,
    _is_span_too_long,
    get_line_spans_within_polygon,
    _cut,
)


def test_get_spanning_line():
    """
    Three colinear segments ⇒ spanning line must run from the
    left-most to the right-most endpoint.
    """
    ml = MultiLineString(
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(3, 3), (4, 4)]),
            LineString([(1, 1), (2, 2)]),
        ]
    )

    expected = LineString([(0, 0), (4, 4)])
    result = get_spanning_line(ml)

    assert result.equals(expected), f"Expected {expected.wkt}, got {result.wkt}"


def test_is_span_too_long():
    base = LineString([(0, 0), (3, 3)])

    # identical lines ⇒ no side is “too long”
    assert _is_span_too_long(base, base) == (False, False)

    # only left endpoint differs
    assert _is_span_too_long(LineString([(0.2, 0.2), (3, 3)]), base) == (True, False)

    # only right endpoint differs
    assert _is_span_too_long(LineString([(0, 0), (3.3, 3.3)]), base) == (False, True)

    # both endpoints differ
    assert _is_span_too_long(LineString([(0.1, 0.1), (3.2, 3.2)]), base) == (True, True)


# Test for first safeguard: retry halving
def test_retry_limit_halved(monkeypatch):
    # stub grow to no-op
    monkeypatch.setattr(gc, "_grow", lambda line, buf, sides: line)
    # stub cut to track calls
    stub_line = LineString([(0, 0), (2, 0)])
    calls = []

    def stub_cut(poly, ext_line, limit):
        calls.append((poly, ext_line, limit))
        return [stub_line]

    monkeypatch.setattr(gc, "_cut", stub_cut)
    # simulate only left endpoint finishing
    monkeypatch.setattr(gc, "_is_span_too_long", lambda span, orig: (True, False))
    # set retry limit
    monkeypatch.setattr(gc, "RETRIES", 5)
    poly = Polygon([(0, -1), (3, -1), (3, 1), (0, 1)])
    line = LineString([(0, 0), (2, 0)])
    result = get_line_spans_within_polygon(poly, line, buffer=0.1, limit=4)
    # should call _cut exactly twice due to halved retries
    assert len(calls) == 2
    # returned cut matches stub
    assert result == [stub_line]


# Test for fallback safeguard when retry limit reached
def test_retry_limit_reached_fallback(monkeypatch):
    # stub grow to no-op
    monkeypatch.setattr(gc, "_grow", lambda line, buf, sides: line)
    # simulate never finishing
    monkeypatch.setattr(gc, "_is_span_too_long", lambda span, orig: (False, False))
    # small retry limit
    monkeypatch.setattr(gc, "RETRIES", 2)
    poly = Polygon([(1, -1), (3, -1), (3, 1), (1, 1)])
    line = LineString([(0, 0), (4, 0)])
    expected = _cut(poly, line, limit=4)
    result = get_line_spans_within_polygon(poly, line, buffer=0.1, limit=4)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r.equals(e)
