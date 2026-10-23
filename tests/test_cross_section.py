"""Benchmarks + documented-behaviour/boundary checks for manifold3d.CrossSection."""
import math

import numpy as np
import pytest
import manifold3d as m

from conftest import approx

Op = m.OpType


@pytest.fixture
def square():
    return m.CrossSection.square((2, 2))


class TestConstructors:
    def test_square(self, benchmark):
        s = benchmark(m.CrossSection.square, (2, 3))
        assert s.area() == approx(6)
        assert s.bounds() == (0, 0, 2, 3)
        assert (s.num_vert(), s.num_contour()) == (4, 1)

    def test_square_center(self):
        assert m.CrossSection.square((2, 4), True).bounds() == (-1, -2, 1, 2)

    @pytest.mark.parametrize("size", [(0, 1), (-1, 1), (1, 0)])
    def test_square_nonpositive_has_no_area(self, size):
        s = m.CrossSection.square(size)
        assert s.area() == 0

    def test_circle(self, benchmark):
        c = benchmark(m.CrossSection.circle, 2.0, 64)
        assert c.num_vert() == 64
        assert c.area() == approx(0.5 * 64 * 4 * math.sin(2 * math.pi / 64))
        assert c.area() == approx(math.pi * 4, rel=0.01)

    @pytest.mark.parametrize("r", [0, -1])
    def test_circle_nonpositive_is_empty(self, r):
        assert m.CrossSection.circle(r).is_empty()

    def test_circle_segments_rounded_not_below_3(self):
        assert m.CrossSection.circle(1, 3).num_vert() >= 3

    def test_empty(self, benchmark):
        e = benchmark(m.CrossSection)
        assert e.is_empty() and e.area() == 0 and e.num_vert() == 0 and e.num_contour() == 0

    def test_from_contours(self, benchmark):
        tri = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        c = benchmark(m.CrossSection, [tri])
        assert c.area() == approx(2)

    def test_from_contours_cw_ignored_by_positive_rule(self):
        cw = np.array([[0, 0], [0, 2], [2, 0]], dtype=float)
        assert m.CrossSection([cw]).is_empty()  # default FillRule.Positive
        assert m.CrossSection([cw], m.FillRule.Negative).area() == approx(2)

    def test_fill_rules(self):
        outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        inner = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)  # same winding
        assert m.CrossSection([outer, inner], m.FillRule.NonZero).area() == approx(16)
        assert m.CrossSection([outer, inner], m.FillRule.Positive).area() == approx(16)
        assert m.CrossSection([outer, inner], m.FillRule.EvenOdd).area() == approx(12)
        assert m.CrossSection([outer, inner], m.FillRule.Negative).is_empty()

    def test_to_polygons(self, benchmark, square):
        polys = benchmark(square.to_polygons)
        assert len(polys) == 1 and np.asarray(polys[0]).shape == (4, 2)

    def test_to_polygons_empty(self):
        assert m.CrossSection().to_polygons() == []


class TestBooleans:
    @pytest.fixture
    def pair(self):
        return m.CrossSection.square((2, 2)), m.CrossSection.square((2, 2)).translate((1, 0))

    def test_add(self, benchmark, pair):
        assert benchmark(lambda: pair[0] + pair[1]).area() == approx(6)

    def test_subtract(self, benchmark, pair):
        r = benchmark(lambda: pair[0] - pair[1])
        assert r.area() == approx(2) and r.bounds() == (0, 0, 1, 2)

    def test_intersect(self, benchmark, pair):
        r = benchmark(lambda: pair[0] ^ pair[1])
        assert r.area() == approx(2) and r.bounds() == (1, 0, 2, 2)

    def test_disjoint_intersection_empty(self, square):
        assert (square ^ square.translate((10, 0))).is_empty()

    def test_hole_creates_two_contours(self):
        r = m.CrossSection.square((4, 4)) - m.CrossSection.square((2, 2)).translate((1, 1))
        assert r.num_contour() == 2 and r.area() == approx(12)

    def test_batch_boolean(self, benchmark):
        xs = [m.CrossSection.square((1, 1)).translate((i * 0.5, 0)) for i in range(10)]
        assert benchmark(m.CrossSection.batch_boolean, xs, Op.Add).area() == approx(5.5)

    def test_batch_boolean_intersect_subtract(self, pair):
        assert m.CrossSection.batch_boolean(list(pair), Op.Intersect).area() == approx(2)
        assert m.CrossSection.batch_boolean(list(pair), Op.Subtract).area() == approx(2)

    def test_batch_boolean_empty_list(self):
        assert m.CrossSection.batch_boolean([], Op.Add).is_empty()

    def test_compose(self, benchmark, square):
        r = benchmark(m.CrossSection.compose, [square, square.translate((5, 0))])
        assert r.area() == approx(8) and r.num_contour() == 2

    def test_decompose(self, benchmark, square):
        c = m.CrossSection.compose([square, square.translate((5, 0)), square.translate((0, 5))])
        parts = benchmark(c.decompose)
        assert len(parts) == 3 and all(p.area() == approx(4) for p in parts)

    def test_decompose_empty(self):
        assert all(p.area() == 0 for p in m.CrossSection().decompose())


class TestHull:
    def test_hull(self, benchmark, square):
        two = square + square.translate((4, 0))
        assert benchmark(two.hull).area() == approx(12)

    def test_hull_empty(self):
        h = m.CrossSection().hull()
        assert h.area() == 0 and h.num_vert() == 0

    def test_hull_points(self, benchmark):
        pts = np.array([[0, 0], [2, 0], [0, 2], [1, 1]], dtype=float)
        assert benchmark(m.CrossSection.hull_points, pts).area() == approx(2)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_hull_points_fewer_than_three_has_no_area(self, n):
        h = m.CrossSection.hull_points(np.random.rand(n, 2))
        assert h.area() == 0 and h.num_vert() == 0

    # Documented as returning an empty CrossSection, but is_empty() is False in 3.5.4.
    @pytest.mark.xfail(reason="docs say empty; is_empty() returns False", strict=False)
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_hull_points_fewer_than_three_is_empty(self, n):
        assert m.CrossSection.hull_points(np.random.rand(n, 2)).is_empty()

    def test_hull_points_collinear_has_no_area(self):
        pts = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        assert m.CrossSection.hull_points(pts).area() == 0

    def test_batch_hull(self, benchmark, square):
        assert benchmark(m.CrossSection.batch_hull, [square, square.translate((2, 0))]).area() == approx(8)

    def test_batch_hull_empty_list(self):
        assert m.CrossSection.batch_hull([]).area() == 0


class TestTransforms:
    def test_translate(self, benchmark, square):
        assert benchmark(square.translate, (1, 2)).bounds() == (1, 2, 3, 4)

    def test_rotate(self, benchmark):
        s = m.CrossSection.square((1, 2))
        r = benchmark(s.rotate, 90)
        assert r.bounds() == pytest.approx((-2, 0, 0, 1), abs=1e-9)
        assert r.area() == approx(2)

    def test_rotate_zero_and_360(self, square):
        assert square.rotate(0).bounds() == (0, 0, 2, 2)
        assert square.rotate(360).bounds() == pytest.approx((0, 0, 2, 2), abs=1e-9)

    def test_scale(self, benchmark, square):
        r = benchmark(square.scale, (2, 3))
        assert r.bounds() == (0, 0, 4, 6) and r.area() == approx(24)

    def test_scale_negative_is_still_valid(self, square):
        r = square.scale((-1, 1))
        assert r.bounds() == (-2, 0, 0, 2) and r.area() == approx(4)

    def test_mirror(self, benchmark, square):
        r = benchmark(square.mirror, (1, 0))
        assert r.bounds() == (-2, 0, 0, 2) and r.area() == approx(4)

    def test_mirror_zero_vector_is_empty(self, square):
        assert square.mirror((0, 0)).is_empty()

    def test_transform(self, benchmark, square):
        mat = np.array([[2, 0, 1], [0, 3, 2]], dtype=float)
        r = benchmark(square.transform, mat)
        assert r.bounds() == (1, 2, 5, 8)

    def test_warp(self, benchmark, square):
        r = benchmark(square.warp, lambda p: (p[0] + 5, p[1]))
        assert r.bounds() == (5, 0, 7, 2)

    # warp_batch leaves the CrossSection unchanged in 3.5.4 (Manifold.warp_batch works).
    @pytest.mark.xfail(reason="CrossSection.warp_batch has no effect", strict=False)
    def test_warp_batch(self, benchmark, square):
        def f(pts):
            pts[:, 0] += 5

        assert benchmark(square.warp_batch, f).bounds() == (5, 0, 7, 2)


class TestOffsetSimplify:
    @pytest.mark.parametrize("jt", [m.JoinType.Round, m.JoinType.Square, m.JoinType.Miter, m.JoinType.Bevel])
    def test_offset_outward(self, benchmark, jt):
        s = m.CrossSection.square((2, 2))
        r = benchmark(s.offset, 1.0, jt)
        assert r.area() > 4
        assert r.bounds() == pytest.approx((-1, -1, 3, 3))

    def test_offset_area_by_join(self):
        s = m.CrossSection.square((2, 2))
        miter = s.offset(1, m.JoinType.Miter).area()
        rnd = s.offset(1, m.JoinType.Round, 2.0, 64).area()
        assert miter == approx(16)
        assert rnd == approx(4 + 8 + math.pi, rel=0.01)
        assert s.offset(1, m.JoinType.Bevel).area() < miter

    def test_offset_inward(self):
        r = m.CrossSection.square((4, 4)).offset(-1, m.JoinType.Miter)
        assert r.area() == approx(4) and r.bounds() == (1, 1, 3, 3)

    def test_offset_zero_is_identity(self, square):
        assert square.offset(0).area() == approx(4)

    def test_offset_to_nothing_is_empty(self, square):
        assert square.offset(-5).is_empty()

    def test_simplify(self, benchmark):
        pts = np.array([[0, 0], [1, 1e-9], [2, 0], [2, 2], [0, 2]], dtype=float)
        c = m.CrossSection([pts])
        r = benchmark(c.simplify, 1e-6)
        assert r.num_vert() <= c.num_vert()
        assert r.area() == approx(4, rel=1e-6)

    def test_simplify_large_epsilon_removes_detail(self):
        c = m.CrossSection.circle(1, 64)
        assert c.simplify(0.5).num_vert() < 64


class TestQueriesAndExtrude:
    def test_area(self, benchmark, square):
        assert benchmark(square.area) == approx(4)

    def test_bounds(self, benchmark, square):
        assert benchmark(square.bounds) == (0, 0, 2, 2)

    def test_bounds_empty(self):
        b = m.CrossSection().bounds()
        assert len(b) == 4

    def test_num_vert_contour(self, benchmark, square):
        assert benchmark(square.num_vert) == 4
        assert square.num_contour() == 1

    def test_is_empty(self, benchmark):
        assert benchmark(m.CrossSection().is_empty) is True

    def test_extrude(self, benchmark, square):
        r = benchmark(square.extrude, 3.0)
        assert r.volume() == approx(12) and r.bounding_box() == (0, 0, 0, 2, 2, 3)

    def test_extrude_nonpositive_height_empty(self, square):
        assert square.extrude(0).is_empty()
