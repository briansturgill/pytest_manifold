"""Benchmarks + documented-behaviour/boundary checks for manifold3d.Manifold."""
import math

import numpy as np
import pytest
import manifold3d as m

from conftest import approx

Op = m.OpType


# ---------------------------------------------------------------- constructors
class TestPrimitives:
    def test_cube(self, benchmark):
        c = benchmark(m.Manifold.cube, (1, 2, 3))
        assert c.bounding_box() == (0, 0, 0, 1, 2, 3)
        assert c.volume() == approx(6)
        assert (c.num_vert(), c.num_tri(), c.num_edge()) == (8, 12, 18)

    def test_cube_default_is_unit_at_origin_corner(self):
        assert m.Manifold.cube().bounding_box() == (0, 0, 0, 1, 1, 1)

    def test_cube_center(self):
        c = m.Manifold.cube((2, 4, 6), True)
        assert c.bounding_box() == (-1, -2, -3, 1, 2, 3)

    @pytest.mark.parametrize("size", [(0, 1, 1), (-1, 1, 1), (1, 1, 0)])
    def test_cube_nonpositive_has_no_volume(self, size):
        c = m.Manifold.cube(size)
        assert c.is_empty() or c.volume() == 0

    def test_cylinder(self, benchmark):
        c = benchmark(m.Manifold.cylinder, 2.0, 1.0, 1.0, 64)
        assert c.volume() == approx(0.5 * 64 * math.sin(2 * math.pi / 64) * 2, rel=1e-6)
        x0, y0, z0, x1, y1, z1 = c.bounding_box()
        assert (z0, z1) == (0, 2)
        assert x1 == approx(1)

    def test_cylinder_cone_when_radius_high_zero(self):
        c = m.Manifold.cylinder(3.0, 1.0, 0.0, 64)
        assert c.volume() == approx(1 / 3 * 3 * 0.5 * 64 * math.sin(2 * math.pi / 64), rel=1e-6)

    def test_cylinder_radius_high_default_means_same_as_low(self):
        a = m.Manifold.cylinder(1, 1, -1, 32)
        b = m.Manifold.cylinder(1, 1, 1, 32)
        assert a.volume() == approx(b.volume())

    def test_cylinder_center(self):
        c = m.Manifold.cylinder(4.0, 1.0, circular_segments=16, center=True)
        assert c.bounding_box()[2] == approx(-2) and c.bounding_box()[5] == approx(2)

    @pytest.mark.parametrize("h,r", [(0, 1), (-1, 1), (1, 0), (1, -1)])
    def test_cylinder_degenerate_is_empty(self, h, r):
        assert m.Manifold.cylinder(h, r).is_empty()

    def test_sphere(self, benchmark):
        s = benchmark(m.Manifold.sphere, 2.0, 64)
        exact = 4 / 3 * math.pi * 8
        assert s.volume() < exact
        assert s.volume() == approx(exact, rel=0.01)
        assert s.surface_area() == approx(4 * math.pi * 4, rel=0.01)
        assert s.genus() == 0

    def test_sphere_segments(self):
        assert m.Manifold.sphere(1, 4).num_tri() == 8
        assert m.Manifold.sphere(1, 8).num_tri() == 32
        # rounded up to a multiple of four
        assert m.Manifold.sphere(1, 5).num_tri() == m.Manifold.sphere(1, 8).num_tri()

    @pytest.mark.parametrize("r", [0, -1])
    def test_sphere_nonpositive_radius_is_empty(self, r):
        assert m.Manifold.sphere(r).is_empty()

    def test_tetrahedron(self, benchmark):
        t = benchmark(m.Manifold.tetrahedron)
        assert (t.num_vert(), t.num_tri()) == (4, 4)
        assert t.bounding_box() == (-1, -1, -1, 1, 1, 1)
        assert t.volume() == approx(8 / 3)

    def test_empty_manifold(self, benchmark):
        e = benchmark(m.Manifold)
        assert e.is_empty()
        assert e.num_vert() == e.num_tri() == 0
        assert e.volume() == 0 and e.surface_area() == 0
        assert e.status() == m.Error.NoError


# ------------------------------------------------------------- extrude/revolve
class TestExtrudeRevolve:
    def test_extrude(self, benchmark):
        sq = m.CrossSection.square((2, 3))
        e = benchmark(m.Manifold.extrude, sq, 5.0)
        assert e.volume() == approx(30)
        assert e.bounding_box() == (0, 0, 0, 2, 3, 5)

    def test_extrude_divisions_add_vertices(self):
        sq = m.CrossSection.square((1, 1))
        assert m.Manifold.extrude(sq, 1, 3).num_vert() > m.Manifold.extrude(sq, 1, 0).num_vert()

    def test_extrude_scale_top_zero_gives_pyramid(self):
        sq = m.CrossSection.square((3, 3))
        p = m.Manifold.extrude(sq, 3, 0, 0, (0, 0))
        assert p.volume() == approx(9)  # 1/3 * 9 * 3

    def test_extrude_scale_top_frustum(self):
        sq = m.CrossSection.square((1, 1))
        e = m.Manifold.extrude(sq, 1, 0, 0, (2, 2))
        assert e.volume() == approx(7 / 3)

    def test_extrude_twist_preserves_volume(self):
        sq = m.CrossSection.square((2, 2), True)
        e = m.Manifold.extrude(sq, 2, 20, 90)
        assert e.volume() == approx(8, rel=0.05)

    @pytest.mark.parametrize("h", [0, -1])
    def test_extrude_nonpositive_height_is_empty(self, h):
        assert m.Manifold.extrude(m.CrossSection.square((1, 1)), h).is_empty()

    def test_extrude_empty_cross_section(self):
        assert m.Manifold.extrude(m.CrossSection(), 1).is_empty()

    def test_revolve(self, benchmark):
        sq = m.CrossSection.square((1, 2)).translate((1, 0))
        r = benchmark(m.Manifold.revolve, sq, 128)
        exact = math.pi * (4 - 1) * 2
        assert r.volume() == approx(exact, rel=0.01)
        assert r.bounding_box()[2] == 0

    def test_revolve_partial_degrees(self):
        sq = m.CrossSection.square((1, 1)).translate((1, 0))
        full = m.Manifold.revolve(sq, 64, 360)
        half = m.Manifold.revolve(sq, 64, 180)
        assert half.volume() == approx(full.volume() / 2, rel=1e-2)

    def test_revolve_empty_cross_section(self):
        assert m.Manifold.revolve(m.CrossSection(), 16).is_empty()

    def test_revolve_cross_section_method(self):
        sq = m.CrossSection.square((1, 1)).translate((1, 0))
        assert sq.revolve(32).volume() == approx(m.Manifold.revolve(sq, 32).volume())


# --------------------------------------------------------------------- booleans
class TestBooleans:
    @pytest.fixture
    def pair(self):
        return m.Manifold.cube((2, 2, 2)), m.Manifold.cube((2, 2, 2)).translate((1, 0, 0))

    def test_add(self, benchmark, pair):
        a, b = pair
        r = benchmark(lambda: a + b)
        assert r.volume() == approx(12)

    def test_subtract(self, benchmark, pair):
        a, b = pair
        r = benchmark(lambda: a - b)
        assert r.volume() == approx(4)
        assert r.bounding_box() == (0, 0, 0, 1, 2, 2)

    def test_intersect(self, benchmark, pair):
        a, b = pair
        r = benchmark(lambda: a ^ b)
        assert r.volume() == approx(4)
        assert r.bounding_box() == (1, 0, 0, 2, 2, 2)

    def test_disjoint_intersection_is_empty(self):
        a = m.Manifold.cube()
        assert (a ^ a.translate((5, 0, 0))).is_empty()

    def test_subtract_self_is_empty(self, cube):
        assert (cube - cube).is_empty()

    def test_add_empty_is_identity(self, cube):
        assert (cube + m.Manifold()).volume() == approx(1)

    def test_touching_faces_union_merges(self):
        a = m.Manifold.cube()
        r = a + a.translate((1, 0, 0))
        assert r.volume() == approx(2) and len(r.decompose()) == 1

    def test_batch_boolean(self, benchmark):
        cubes = [m.Manifold.cube().translate((i * 0.5, 0, 0)) for i in range(10)]
        r = benchmark(m.Manifold.batch_boolean, cubes, Op.Add)
        assert r.volume() == approx(5.5)

    def test_batch_boolean_ops(self):
        a = m.Manifold.cube((2, 2, 2))
        b = a.translate((1, 0, 0))
        assert m.Manifold.batch_boolean([a, b], Op.Intersect).volume() == approx(4)
        assert m.Manifold.batch_boolean([a, b], Op.Subtract).volume() == approx(4)

    def test_batch_boolean_empty_list(self):
        assert m.Manifold.batch_boolean([], Op.Add).is_empty()

    def test_batch_boolean_single(self, cube):
        assert m.Manifold.batch_boolean([cube], Op.Add).volume() == approx(1)

    def test_compose(self, benchmark):
        a = m.Manifold.cube()
        r = benchmark(m.Manifold.compose, [a, a.translate((5, 0, 0))])
        assert r.volume() == approx(2)
        assert len(r.decompose()) == 2

    def test_decompose(self, benchmark):
        a = m.Manifold.cube()
        comp = m.Manifold.compose([a, a.translate((5, 0, 0)), a.translate((0, 5, 0))])
        parts = benchmark(comp.decompose)
        assert len(parts) == 3
        assert all(p.volume() == approx(1) for p in parts)

    def test_decompose_empty(self):
        assert m.Manifold().decompose() == []


# -------------------------------------------------------------------------- hull
class TestHull:
    def test_hull(self, benchmark):
        two = m.Manifold.cube() + m.Manifold.cube().translate((3, 0, 0))
        h = benchmark(two.hull)
        assert h.volume() == approx(4)

    def test_hull_of_convex_is_itself(self, cube):
        assert cube.hull().volume() == approx(1)

    def test_hull_empty(self):
        assert m.Manifold().hull().is_empty()

    def test_hull_points(self, benchmark):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        h = benchmark(m.Manifold.hull_points, pts)
        assert h.volume() == approx(1 / 6)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_hull_points_fewer_than_four_has_no_volume(self, n):
        assert m.Manifold.hull_points(np.random.rand(n, 3)).volume() == pytest.approx(0, abs=1e-12)

    # Documented as returning an empty Manifold, but is_empty() is False in 3.5.4.
    @pytest.mark.xfail(reason="docs say empty; is_empty() returns False", strict=False)
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_hull_points_fewer_than_four_is_empty(self, n):
        assert m.Manifold.hull_points(np.random.rand(n, 3)).is_empty()

    def test_hull_points_coplanar_has_no_volume(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [.5, .5, 0]], dtype=float)
        assert m.Manifold.hull_points(pts).volume() == pytest.approx(0, abs=1e-12)

    def test_batch_hull(self, benchmark):
        c = m.Manifold.cube()
        h = benchmark(m.Manifold.batch_hull, [c, c.translate((2, 0, 0))])
        assert h.volume() == approx(3)

    def test_batch_hull_empty_list(self):
        assert m.Manifold.batch_hull([]).is_empty()


# ----------------------------------------------------------------- transformation
class TestTransforms:
    def test_translate(self, benchmark, cube):
        r = benchmark(cube.translate, (1, 2, 3))
        assert r.bounding_box() == (1, 2, 3, 2, 3, 4)

    def test_translate_does_not_mutate(self, cube):
        cube.translate((9, 9, 9))
        assert cube.bounding_box() == (0, 0, 0, 1, 1, 1)

    def test_rotate(self, benchmark):
        c = m.Manifold.cube((1, 2, 3))
        r = benchmark(c.rotate, (0, 0, 90))
        assert r.bounding_box() == pytest.approx((-2, 0, 0, 0, 1, 3), abs=1e-6)
        assert r.volume() == approx(6)

    def test_rotate_zero_is_identity(self, cube):
        assert cube.rotate((0, 0, 0)).bounding_box() == pytest.approx((0, 0, 0, 1, 1, 1))

    def test_rotate_360_returns(self, cube):
        assert cube.rotate((360, 0, 0)).bounding_box() == pytest.approx((0, 0, 0, 1, 1, 1), abs=1e-6)

    def test_scale(self, benchmark, cube):
        r = benchmark(cube.scale, (2, 3, 4))
        assert r.bounding_box() == (0, 0, 0, 2, 3, 4)
        assert r.volume() == approx(24)

    def test_scale_negative_keeps_valid_orientation(self, cube):
        r = cube.scale((-1, 1, 1))
        assert r.bounding_box() == (-1, 0, 0, 0, 1, 1)
        assert r.volume() == approx(1)

    def test_scale_zero_collapses(self, cube):
        r = cube.scale((0, 1, 1))
        assert r.is_empty() or r.volume() == 0

    def test_mirror(self, benchmark):
        c = m.Manifold.cube().translate((1, 0, 0))
        r = benchmark(c.mirror, (1, 0, 0))
        assert r.bounding_box() == (-2, 0, 0, -1, 1, 1)
        assert r.volume() == approx(1)

    def test_mirror_zero_vector_is_empty(self, cube):
        assert cube.mirror((0, 0, 0)).is_empty()

    def test_mirror_normal_is_normalised(self):
        c = m.Manifold.cube().translate((1, 0, 0))
        assert c.mirror((5, 0, 0)).bounding_box() == (-2, 0, 0, -1, 1, 1)

    def test_transform(self, benchmark, cube):
        mat = [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]]
        r = benchmark(cube.transform, np.array(mat, dtype=float))
        assert r.bounding_box() == (5, 6, 7, 6, 7, 8)

    def test_transform_matches_scale(self, cube):
        mat = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0]], dtype=float)
        assert cube.transform(mat).volume() == approx(24)

    def test_warp(self, benchmark, cube):
        r = benchmark(cube.warp, lambda v: (v[0] + 10, v[1], v[2]))
        assert r.bounding_box() == (10, 0, 0, 11, 1, 1)

    def test_warp_batch(self, benchmark, cube):
        r = benchmark(cube.warp_batch, lambda pts: pts + np.array([10.0, 0.0, 0.0]))
        assert r.bounding_box() == (10, 0, 0, 11, 1, 1)


# ---------------------------------------------------------------------- queries
class TestQueries:
    def test_bounding_box(self, benchmark, cube):
        assert benchmark(cube.bounding_box) == (0, 0, 0, 1, 1, 1)

    def test_volume_surface_area(self, benchmark):
        c = m.Manifold.cube((1, 2, 3))
        assert benchmark(c.volume) == approx(6)
        assert c.surface_area() == approx(22)

    def test_surface_area_benchmark(self, benchmark, big_sphere):
        assert benchmark(big_sphere.surface_area) > 0

    def test_counts(self, benchmark, cube):
        assert benchmark(cube.num_vert) == 8
        assert cube.num_tri() == 12
        assert cube.num_edge() == 18
        assert cube.num_vert() - cube.num_edge() + cube.num_tri() == 2  # Euler

    def test_num_prop(self, benchmark, cube):
        assert benchmark(cube.num_prop) == 0
        assert cube.num_prop_vert() == cube.num_vert()
        assert cube.set_properties(2, lambda p, o: [1.0, 2.0]).num_prop() == 2

    def test_genus(self, benchmark):
        torus = m.Manifold.revolve(m.CrossSection.circle(1, 16).translate((3, 0)), 16)
        assert benchmark(torus.genus) == 1
        assert m.Manifold.cube().genus() == 0

    def test_is_empty(self, benchmark):
        assert benchmark(m.Manifold().is_empty) is True
        assert m.Manifold.cube().is_empty() is False

    def test_status(self, benchmark, cube):
        assert benchmark(cube.status) == m.Error.NoError

    def test_status_reports_error_for_bad_mesh(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        t = np.array([[0, 1, 2]], dtype=np.uint32)
        bad = m.Manifold(m.Mesh(v, t))
        assert bad.is_empty()
        assert bad.status() == m.Error.NotManifold

    def test_original_id_and_as_original(self, benchmark):
        c = m.Manifold.cube()
        oid = c.original_id()
        assert oid >= 0
        moved = c.translate((1, 0, 0))
        r = benchmark(moved.as_original)
        assert r.original_id() != oid
        assert r.bounding_box() == (1, 0, 0, 2, 1, 1)

    def test_reserve_ids(self, benchmark):
        a = benchmark(m.Manifold.reserve_ids, 3)
        b = m.Manifold.reserve_ids(1)
        assert b >= a + 3

    def test_tolerance(self, benchmark, cube):
        t = benchmark(cube.get_tolerance)
        assert t > 0
        assert cube.set_tolerance(0.5).get_tolerance() >= 0.5

    def test_min_gap(self, benchmark):
        a = m.Manifold.cube()
        b = a.translate((3, 0, 0))
        assert benchmark(a.min_gap, b, 10) == approx(2)

    def test_min_gap_beyond_search_length(self):
        a = m.Manifold.cube()
        assert a.min_gap(a.translate((30, 0, 0)), 1) == approx(1)

    def test_min_gap_overlapping_is_zero(self, cube):
        assert cube.min_gap(cube, 1) == 0

    def test_ray_cast(self, benchmark, cube):
        hits = benchmark(cube.ray_cast, (0.5, 0.5, -1), (0.5, 0.5, 5))
        assert len(hits) == 2
        assert hits[0].distance < hits[1].distance
        assert tuple(hits[0].position) == pytest.approx((0.5, 0.5, 0.0))
        assert tuple(hits[0].normal) == pytest.approx((0, 0, -1))
        assert tuple(hits[1].position) == pytest.approx((0.5, 0.5, 1.0))

    def test_ray_cast_miss_and_short(self, cube):
        assert cube.ray_cast((5, 5, -1), (5, 5, 5)) == []
        assert cube.ray_cast((0.5, 0.5, -1), (0.5, 0.5, -0.5)) == []


# -------------------------------------------------------------- slice / project
class TestSliceProject:
    def test_slice(self, benchmark):
        s = m.Manifold.cube((2, 3, 4))
        cs = benchmark(s.slice, 1.0)
        assert cs.area() == approx(6)

    @pytest.mark.parametrize("h", [-0.1, 4.1])
    def test_slice_outside_is_empty(self, h):
        assert m.Manifold.cube((2, 3, 4)).slice(h).is_empty()

    def test_project(self, benchmark):
        c = m.Manifold.cube((2, 3, 4))
        p = benchmark(c.project)
        assert p.area() == approx(6)
        assert p.bounds() == (0, 0, 2, 3)

    def test_project_empty(self):
        assert m.Manifold().project().is_empty()


# ----------------------------------------------------------------------- splitting
class TestSplit:
    def test_split(self, benchmark):
        a = m.Manifold.cube((2, 2, 2))
        cutter = m.Manifold.cube((1, 2, 2))
        inside, outside = benchmark(a.split, cutter)
        assert inside.volume() == approx(4)
        assert outside.volume() == approx(4)

    def test_split_by_plane(self, benchmark):
        c = m.Manifold.cube((2, 2, 2))
        top, bottom = benchmark(c.split_by_plane, (0, 0, 1), 0.5)
        assert top.volume() == approx(6)
        assert bottom.volume() == approx(2)

    def test_split_by_plane_outside(self, cube):
        top, bottom = cube.split_by_plane((0, 0, 1), 5)
        assert top.is_empty() and bottom.volume() == approx(1)

    def test_trim_by_plane(self, benchmark):
        c = m.Manifold.cube((2, 2, 2))
        r = benchmark(c.trim_by_plane, (0, 0, 1), 0.5)
        assert r.volume() == approx(6)
        assert r.bounding_box()[2] == approx(0.5)


# ------------------------------------------------------------------------ refining
class TestRefine:
    def test_refine(self, benchmark, cube):
        r = benchmark(cube.refine, 3)
        assert r.num_tri() == cube.num_tri() * 9
        assert r.volume() == approx(1)

    def test_refine_1_is_noop(self, cube):
        assert cube.refine(1).num_tri() == cube.num_tri()

    def test_refine_to_length(self, benchmark, cube):
        r = benchmark(cube.refine_to_length, 0.25)
        assert r.num_tri() > cube.num_tri()

    def test_refine_to_tolerance(self, benchmark):
        s = m.Manifold.sphere(1, 16).smooth_out()
        r = benchmark(s.refine_to_tolerance, 0.01)
        assert r.num_tri() >= s.num_tri()

    def test_simplify(self, benchmark):
        c = m.Manifold.cube().refine(4)
        r = benchmark(c.simplify, 1e-5)
        assert r.num_tri() <= c.num_tri()
        assert r.volume() == approx(1)


# ------------------------------------------------------------------------ smoothing
class TestSmooth:
    def test_smooth_out(self, benchmark):
        c = m.Manifold.cube((2, 2, 2), True)
        s = benchmark(c.smooth_out)
        assert s.num_tri() == c.num_tri()  # smoothing only adds tangents
        r = s.refine(8)
        assert 4 < r.volume() < 8
        assert r.bounding_box() == pytest.approx((-1, -1, -1, 1, 1, 1))

    def test_smooth_out_matches_defaults(self):
        c = m.Manifold.cube((2, 2, 2), True)
        assert c.smooth_out(52.5, 0).refine(8).volume() == approx(c.smooth_out().refine(8).volume())

    def test_smooth_by_normals(self, benchmark):
        s = m.Manifold.sphere(1, 16).calculate_normals(0, 60)
        r = benchmark(s.smooth_by_normals, 0)
        assert r.refine(4).volume() == pytest.approx(4 / 3 * math.pi, rel=0.03)

    def test_calculate_normals(self, benchmark, cube):
        r = benchmark(cube.calculate_normals, 0, 52.5)
        assert r.num_prop() >= 3
        assert r.volume() == approx(1)

    def test_calculate_curvature(self, benchmark, big_sphere):
        r = benchmark(big_sphere.calculate_curvature, 0, 1)
        assert r.num_prop() >= 2
        props = np.asarray(r.to_mesh().vert_properties)
        # Gaussian curvature = 1/r^2, mean curvature = 2/r (sum of principal curvatures)
        assert np.median(props[:, 3]) == pytest.approx(1, rel=0.1)
        assert np.median(props[:, 4]) == pytest.approx(2, rel=0.1)

    def test_smooth_from_mesh(self, benchmark, cube):
        mesh = cube.to_mesh()
        r = benchmark(m.Manifold.smooth, mesh)
        assert not r.is_empty()
        assert r.status() == m.Error.NoError

    def test_smooth_with_sharpened_edges(self, cube):
        r = m.Manifold.smooth(cube.to_mesh(), [0, 1], [0.0, 0.0])
        assert not r.is_empty()


# ------------------------------------------------------------- properties / levelset
class TestPropertiesAndLevelSet:
    def test_set_properties(self, benchmark, cube):
        r = benchmark(cube.set_properties, 1, lambda pos, old: [pos[0]])
        assert r.num_prop() == 1
        vp = np.asarray(r.to_mesh().vert_properties)
        assert np.allclose(vp[:, 3], vp[:, 0])

    def test_set_properties_zero_removes(self, cube):
        r = cube.set_properties(1, lambda p, o: [1.0]).set_properties(0, lambda p, o: [])
        assert r.num_prop() == 0

    def test_level_set(self, benchmark):
        def f(x, y, z):
            return 1 - math.sqrt(x * x + y * y + z * z)

        r = benchmark(m.Manifold.level_set, f, [-2, -2, -2, 2, 2, 2], 0.2)
        assert r.volume() == approx(4 / 3 * math.pi, rel=0.1)
        assert all(abs(abs(v) - 1) < 0.25 for v in r.bounding_box())

    def test_level_set_level_shifts_surface(self):
        def f(x, y, z):
            return 2 - math.sqrt(x * x + y * y + z * z)

        r = m.Manifold.level_set(f, [-3, -3, -3, 3, 3, 3], 0.25, level=1.0)
        assert r.volume() == approx(4 / 3 * math.pi, rel=0.15)

    def test_level_set_nothing_inside_is_empty(self):
        r = m.Manifold.level_set(lambda x, y, z: -1.0, [-1, -1, -1, 1, 1, 1], 0.5)
        assert r.is_empty()


# ----------------------------------------------------------------------- minkowski
class TestMinkowski:
    def test_minkowski_sum(self, benchmark):
        a = m.Manifold.cube((2, 2, 2), True)
        b = m.Manifold.cube((1, 1, 1), True)
        r = benchmark(a.minkowski_sum, b)
        assert r.bounding_box() == pytest.approx((-1.5, -1.5, -1.5, 1.5, 1.5, 1.5))
        assert r.volume() == approx(27)

    def test_minkowski_difference(self, benchmark):
        a = m.Manifold.cube((3, 3, 3), True)
        b = m.Manifold.cube((1, 1, 1), True)
        r = benchmark(a.minkowski_difference, b)
        assert r.bounding_box() == pytest.approx((-1, -1, -1, 1, 1, 1))
        assert r.volume() == approx(8)


# ------------------------------------------------------------------ mesh round trip
class TestMeshIO:
    def test_to_mesh(self, benchmark, cube):
        mesh = benchmark(cube.to_mesh)
        assert np.asarray(mesh.vert_properties).shape[1] == 3
        assert np.asarray(mesh.tri_verts).shape == (12, 3)

    def test_to_mesh64(self, benchmark, cube):
        mesh = benchmark(cube.to_mesh64)
        assert np.asarray(mesh.vert_properties).dtype == np.float64
        assert np.asarray(mesh.tri_verts).shape == (12, 3)

    def test_mesh_roundtrip(self, benchmark, big_sphere):
        mesh = big_sphere.to_mesh()
        r = benchmark(m.Manifold, mesh)
        assert r.volume() == approx(big_sphere.volume(), rel=1e-5)
        assert r.num_tri() == big_sphere.num_tri()

    def test_mesh64_roundtrip(self, big_sphere):
        r = m.Manifold(big_sphere.to_mesh64())
        assert r.volume() == approx(big_sphere.volume())

    def test_with_context(self, benchmark, cube):
        ctx = m.ExecutionContext()
        r = benchmark(cube.with_context, ctx)
        assert r.volume() == approx(1)
        assert not ctx.cancelled()
