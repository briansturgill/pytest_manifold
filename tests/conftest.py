import numpy as np
import pytest
import manifold3d as m


@pytest.fixture(autouse=True)
def reset_circular_defaults():
    """Restore global circle-quality settings after every test."""
    yield
    m.set_circular_segments(0)
    m.set_min_circular_angle(10.0)
    m.set_min_circular_edge_length(1.0)


@pytest.fixture
def cube():
    return m.Manifold.cube((1, 1, 1))


@pytest.fixture
def big_sphere():
    return m.Manifold.sphere(1.0, 64)


def approx(x, rel=1e-6, abs=1e-9):
    return pytest.approx(x, rel=rel, abs=abs)
