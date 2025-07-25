import pytest
from shapely.geometry import LineString

from inference.grow_cut import (
    _grow,
    _is_span_too_long,
    TOLERANCE,
    _split_line_by_heading,
)


def test_grow_zero_buffer():
    line = LineString([(0, 0), (1, 0)])
    # With zero buffer, line should not change regardless of sides
    assert list(_grow(line, buffer=0, sides=(True, True)).coords) == list(line.coords)
    assert list(_grow(line, buffer=0, sides=(True, False)).coords) == list(line.coords)
    assert list(_grow(line, buffer=0, sides=(False, True)).coords) == list(line.coords)


def test_grow_single_side():
    line = LineString([(0, 0), (1, 0)])
    # Grow left side only
    new_left = _grow(line, buffer=0.1, sides=(True, False))
    # Expect start moves by -0.1, end unchanged
    assert pytest.approx(new_left.coords[0][0], rel=1e-6) == -0.1
    assert new_left.coords[1] == (1.0, 0.0)

    # Grow right side only
    new_right = _grow(line, buffer=0.1, sides=(False, True))
    assert new_right.coords[0] == (0.0, 0.0)
    assert pytest.approx(new_right.coords[1][0], rel=1e-6) == 1.1


def test_grow_both_sides():
    line = LineString([(0, 0), (1, 0)])
    new_line = _grow(line, buffer=0.1, sides=(True, True))
    # Both ends moved by 0.1
    assert pytest.approx(new_line.coords[0][0], rel=1e-6) == -0.1
    assert pytest.approx(new_line.coords[1][0], rel=1e-6) == 1.1


def test_is_span_too_long_orientation():
    original = LineString([(0, 0), (1, 0)])
    # No extension
    span_same = LineString([(0, 0), (1, 0)])
    lf, rf = _is_span_too_long(span_same, original)
    assert lf is False and rf is False

    # Extend right beyond tolerance
    extended_dist = 1 + TOLERANCE + 0.2
    span_ext = LineString([(0, 0), (extended_dist, 0)])
    lf2, rf2 = _is_span_too_long(span_ext, original)
    assert lf2 is False and rf2 is True

    # Swapped endpoints order
    span_swapped = LineString([(extended_dist, 0), (0, 0)])
    lf3, rf3 = _is_span_too_long(span_swapped, original)
    assert lf3 is False and rf3 is True


# unit tests for _split_line_by_heading helper
def test_split_line_by_heading_two_segments():
    line = LineString([(0, 0), (1, 0), (2, 0), (2, 1)])
    segments = _split_line_by_heading(line, angle_tol_deg=10)
    assert len(segments) == 2
    expected1 = LineString([(0, 0), (2, 0)])
    expected2 = LineString([(2, 0), (2, 1)])
    assert segments[0].equals(expected1), f"First segment mismatch: {segments[0].wkt}"
    assert segments[1].equals(expected2), f"Second segment mismatch: {segments[1].wkt}"


def test_split_line_by_heading_no_split():
    line = LineString([(0, 0), (1, 0), (2, 0.1), (3, 0.2)])
    segments = _split_line_by_heading(line, angle_tol_deg=10)
    assert len(segments) == 1
    assert list(segments[0].coords) == list(line.coords)
