from shapely.geometry import LineString, MultiLineString

from inference.grow_cut import get_spanning_line, _is_span_too_long


def test_get_spanning_line():
    """
    Three colinear segments ⇒ spanning line must run from the
    left-most to the right-most endpoint.
    """
    ml = MultiLineString([
        LineString([(0, 0), (1, 1)]),
        LineString([(3, 3), (4, 4)]),
        LineString([(1, 1), (2, 2)]),
    ])

    expected = LineString([(0, 0), (4, 4)])
    result   = get_spanning_line(ml)

    assert result.equals(expected), (
        f"Expected {expected.wkt}, got {result.wkt}"
    )

def test_is_span_too_long():
    base = LineString([(0, 0), (3, 3)])

    # identical lines ⇒ no side is “too long”
    assert _is_span_too_long(base, base) == (False, False)

    # only left endpoint differs
    assert _is_span_too_long(LineString([(0.2, 0.2), (3, 3)]), base) == (True,  False)

    # only right endpoint differs
    assert _is_span_too_long(LineString([(0, 0), (3.3, 3.3)]), base) == (False, True)

    # both endpoints differ
    assert _is_span_too_long(LineString([(0.1, 0.1), (3.2, 3.2)]), base) == (True,  True)
