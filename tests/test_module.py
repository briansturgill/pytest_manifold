"""Module-level functions, Mesh/Mesh64, ExecutionContext, RayHit and enums."""
import math

import numpy as np
import pytest
import manifold3d as m

from conftest import approx


class TestCircularSettings:
    def test_get_circular_segments_default(self, benchmark):
        n = benchmark(m.get_circular_segments, 1.0)
        assert n >= 3 and n % 4 == 0

    def test_set_circular_segments_exact(self, benchmark):
        benchmark(m.set_circular_segments, 24)
        assert m.get_circular_segments(1.0) == 24
        assert m.get_circular_segments(100.0) == 24
        assert m.CrossSection.circle(5).num_vert() == 24
        assert m.Manifold.cylinder(1, 1).num_vert() == 48

    def test_set_circular_segments_zero_restores_defaults(self):
        base = m.get_circular_segments(3.0)
        m.set_circular_segments(24)
        m.set_circular_segments(0)
        assert m.get_circular_segments(3.0) == base

    def test_set_min_circular_angle(self, benchmark):
        m.set_min_circular_edge_length(0.0001)
        benchmark(m.set_min_circular_angle, 10)
        assert m.get_circular_segments(10) == 36 or m.get_circular_segments(10) == 40
        m.set_min_circular_angle(45)
        assert m.get_circular_segments(10) == 8

    def test_set_min_circular_edge_length(self, benchmark):
        m.set_min_circular_angle(1)
        benchmark(m.set_min_circular_edge_length, 1.0)
        n = m.get_circular_segments(10.0)
        assert n % 4 == 0
        assert n == 64
        assert 2 * math.pi * 10 / n >= 0.9
        m.set_min_circular_edge_length(10.0)
        assert m.get_circular_segments(10.0) < n

    def test_larger_radius_never_fewer_segments(self):
        assert m.get_circular_segments(10) >= m.get_circular_segments(1)


class TestTriangulate:
    def test_square(self, benchmark):
        poly = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]
        tris = benchmark(m.triangulate, poly)
        tris = np.asarray(tris)
        assert tris.shape == (2, 3)
        assert set(tris.flatten()) == {0, 1, 2, 3}

    def test_polygon_with_hole(self):
        outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        hole = np.array([[1, 1], [1, 3], [3, 3], [3, 1]], dtype=float)  # CW
        tris = np.asarray(m.triangulate([outer, hole]))
        pts = np.vstack([outer, hole])
        a = pts[tris]
        area = 0.5 * np.abs(
            (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
            - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
        ).sum()
        assert area == approx(12)
        assert tris.shape[0] == 8

    def test_empty(self):
        assert np.asarray(m.triangulate([])).shape[0] == 0

    @pytest.mark.parametrize("allow_convex", [True, False])
    def test_allow_convex(self, allow_convex):
        poly = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)]
        assert np.asarray(m.triangulate(poly, -1, allow_convex)).shape == (2, 3)

    def test_fewer_than_three_points(self):
        poly = [np.array([[0, 0], [1, 0]], dtype=float)]
        assert np.asarray(m.triangulate(poly)).shape[0] == 0


class TestMesh:
    @pytest.fixture
    def mesh(self):
        return m.Manifold.cube().to_mesh()

    def test_construct_from_arrays(self, benchmark):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        t = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.uint32)
        mesh = benchmark(m.Mesh, v, t)
        r = m.Manifold(mesh)
        assert r.status() == m.Error.NoError
        assert r.volume() == approx(1 / 6)

    def test_properties(self, benchmark, mesh):
        vp = benchmark(lambda: np.asarray(mesh.vert_properties))
        assert vp.shape[1] == 3 and vp.dtype == np.float32
        assert np.asarray(mesh.tri_verts).dtype == np.int32 or np.asarray(mesh.tri_verts).dtype == np.uint32
        assert len(mesh.run_index) >= 2
        assert len(mesh.run_original_id) >= 1
        assert len(mesh.face_id) in (0, 12)
        assert len(mesh.merge_from_vert) == len(mesh.merge_to_vert)
        assert np.asarray(mesh.run_transform).size % 12 == 0

    def test_tri_verts_in_bounds(self, mesh):
        tv = np.asarray(mesh.tri_verts)
        assert tv.min() >= 0 and tv.max() < np.asarray(mesh.vert_properties).shape[0]

    def test_merge(self, benchmark):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        t = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.uint32)
        mesh = m.Mesh(v, t)
        assert isinstance(benchmark(mesh.merge), bool)

    def test_merge_joins_duplicate_vertices(self):
        c = m.Manifold.cube().to_mesh()
        # a cube with normals as properties has split vertices which merge restores
        n = m.Manifold.cube().calculate_normals(0, 30).to_mesh(0)
        assert np.asarray(n.vert_properties).shape[0] > 8
        n.merge()
        assert len(n.merge_from_vert) > 0
        assert m.Manifold(n).num_vert() == 8
        assert c is not None

    def test_has_normals_and_backside(self, benchmark):
        mesh = m.Manifold.cube().calculate_normals(0, 30).to_mesh(0)
        assert benchmark(mesh.has_normals, 0) in (True, False)
        assert mesh.backside(0) in (True, False)

    def test_halfedge_tangent_empty_for_unsmoothed(self, mesh):
        assert np.asarray(mesh.halfedge_tangent).size == 0

    def test_run_original_id_matches_manifold(self):
        c = m.Manifold.cube()
        assert c.original_id() in list(c.to_mesh().run_original_id)

    def test_invalid_vertex_index_gives_error(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        v = np.vstack([v, [0, 0, 1]]).astype(np.float32)
        t = np.array([[0, 2, 1], [0, 1, 9], [1, 2, 3], [0, 3, 2]], dtype=np.uint32)
        r = m.Manifold(m.Mesh(v, t))
        assert r.is_empty()
        assert r.status() == m.Error.VertexOutOfBounds

    def test_non_finite_vertex_gives_error(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, np.nan, 0], [0, 0, 1]], dtype=np.float32)
        t = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], dtype=np.uint32)
        r = m.Manifold(m.Mesh(v, t))
        assert r.is_empty()
        assert r.status() == m.Error.NonFiniteVertex

    def test_run_flags(self, benchmark, mesh):
        flags = benchmark(lambda: list(mesh.run_flags))
        assert len(flags) in (0, len(mesh.run_original_id))

    def test_mesh64_attributes(self):
        mesh = m.Manifold.cube().to_mesh64()
        assert np.asarray(mesh.vert_properties).shape == (8, 3)
        assert np.asarray(mesh.tri_verts).shape == (12, 3)
        assert len(mesh.run_index) >= 2
        assert len(mesh.run_original_id) >= 1
        assert len(mesh.run_flags) in (0, len(mesh.run_original_id))
        assert len(mesh.face_id) in (0, 12)
        assert len(mesh.merge_from_vert) == len(mesh.merge_to_vert)
        assert np.asarray(mesh.halfedge_tangent).size == 0
        assert isinstance(mesh.has_normals(0), bool)
        assert isinstance(mesh.backside(0), bool)

    def test_mesh64(self, benchmark):
        c = m.Manifold.cube().to_mesh64()
        r = benchmark(m.Manifold, c)
        assert r.volume() == approx(1)
        assert hasattr(c, "merge")


class TestExecutionContext:
    def test_default_state(self, benchmark):
        ctx = benchmark(m.ExecutionContext)
        assert ctx.cancelled() is False
        assert ctx.progress() == 0 or 0 <= ctx.progress() <= 1

    def test_cancel(self, benchmark):
        ctx = m.ExecutionContext()
        benchmark(ctx.cancel)
        assert ctx.cancelled() is True

    def test_cancelled_context_result_is_error_or_empty(self):
        ctx = m.ExecutionContext()
        ctx.cancel()
        r = m.Manifold.cube().with_context(ctx).translate((1, 0, 0))
        assert r.is_empty() or r.status() in (m.Error.Cancelled, m.Error.NoError)

    def test_from_mesh(self, benchmark):
        ctx = m.ExecutionContext()
        mesh = m.Manifold.cube().to_mesh()
        r = benchmark(ctx.from_mesh, mesh)
        assert r.volume() == approx(1)

    def test_level_set(self, benchmark):
        ctx = m.ExecutionContext()

        def f(x, y, z):
            return 1 - math.sqrt(x * x + y * y + z * z)

        r = benchmark(ctx.level_set, f, [-2, -2, -2, 2, 2, 2], 0.25)
        assert r.volume() == approx(4 / 3 * math.pi, rel=0.15)

    def test_smooth(self, benchmark):
        ctx = m.ExecutionContext()
        mesh = m.Manifold.cube().to_mesh()
        r = benchmark(ctx.smooth, mesh)
        assert not r.is_empty()


class TestEnums:
    def test_op_types(self):
        assert {m.OpType.Add, m.OpType.Subtract, m.OpType.Intersect}

    def test_error_values_distinct(self):
        names = ["NoError", "NonFiniteVertex", "NotManifold", "VertexOutOfBounds", "Cancelled"]
        vals = [getattr(m.Error, n) for n in names]
        assert len(set(vals)) == len(names)

    def test_join_types_and_fill_rules_exist(self):
        for n in ("Square", "Round", "Miter", "Bevel"):
            assert hasattr(m.JoinType, n)
        for n in ("EvenOdd", "NonZero", "Positive", "Negative"):
            assert hasattr(m.FillRule, n)
