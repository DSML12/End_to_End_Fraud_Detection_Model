"""PSI: ~0 on identical distributions, material on a clear shift."""
import numpy as np

from src.monitoring.drift import band, bin_edges, psi


def test_psi_zero_on_identical():
    x = np.random.RandomState(0).normal(size=5000)
    edges = bin_edges(x)
    assert psi(x, x, edges) < 0.01
    assert band(psi(x, x, edges)) == "stable"


def test_psi_material_on_shift():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 5000)
    act = rng.normal(2, 1, 5000)  # mean shifted by 2 sigma
    edges = bin_edges(ref)
    assert psi(ref, act, edges) > 0.25
    assert band(psi(ref, act, edges)) == "material"
