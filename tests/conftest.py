import pytest

from ccr import FXModel, HullWhite, ZeroCurve, simulate

CURVE = ZeroCurve((0.25, 1, 2, 5, 10, 20), (0.030, 0.032, 0.034, 0.036, 0.038, 0.039))


@pytest.fixture(scope="session")
def curve() -> ZeroCurve:
    return CURVE


@pytest.fixture(scope="session")
def hw(curve) -> HullWhite:
    return HullWhite(curve, a=0.03, sigma=0.01)


@pytest.fixture(scope="session")
def sim(hw):
    """Shared 10y simulation with FX, weekly grid."""
    return simulate(hw, 10.0, n_paths=8_000, fx=FXModel(1.10, 0.10, 0.02, -0.2), seed=7)
