"""Tests for mumot.continuation (equilibrium continuation via bundled PyDSTool)."""
import numpy as np
import pytest
import sympy

from mumot.continuation import equilibrium_continuation


def test_fold_normal_form():
    """dx/dt = p - x**2 has limit points at p = 0 only."""
    x, p = sympy.symbols('x p')
    cont = equilibrium_continuation({x: p - x**2}, {p: 1.0}, p, max_num_points=200)
    branch = cont.from_state({x: 1.0})
    assert branch is not None
    assert branch.failures == []
    # all points are equilibria
    np.testing.assert_allclose(branch.parameter, branch.states[x]**2, atol=1e-6)
    folds = branch.special('LP')
    assert len(folds) == 1
    assert folds[0].parameter == pytest.approx(0, abs=1e-6)
    assert folds[0].state[x] == pytest.approx(0, abs=1e-3)
    # stable (x > 0) and unstable (x < 0) halves
    eig = branch.eigenvalues[:, 0].real
    assert np.all(eig[branch.states[x] > 0.01] < 0)
    assert np.all(eig[branch.states[x] < -0.01] > 0)


def test_pitchfork_branch_switching():
    """dx/dt = p*x - x**3, dy/dt = -y: branch point at p = 0 on x = 0.

    Switching branch there must give the x**2 = p parabola, not the trivial branch.
    """
    x, y, p = sympy.symbols('x y p')
    cont = equilibrium_continuation({x: p * x - x**3, y: -y}, {p: -1.0}, p,
                                    max_num_points=100)
    trivial = cont.from_state({x: 0.0, y: 0.0})
    np.testing.assert_allclose(trivial.states[x], 0, atol=1e-8)
    branch_points = trivial.special('BP')
    assert len(branch_points) == 1
    assert branch_points[0].parameter == pytest.approx(0, abs=1e-6)

    new = cont.from_branch_point(branch_points[0])
    assert new is not None
    assert new.failures == []
    np.testing.assert_allclose(new.states[x]**2, new.parameter, atol=1e-6)
    # both halves of the parabola are followed
    assert new.states[x].max() > 0.5
    assert new.states[x].min() < -0.5


def test_symbol_names_are_isolated_from_pydstool():
    """Symbol names that PyDSTool treats specially (e.g. ``gamma``) are fine."""
    gamma, beta, x = sympy.symbols('gamma beta x')
    cont = equilibrium_continuation({x: gamma - beta * x}, {gamma: 1.0, beta: 2.0}, gamma,
                                    max_num_points=20)
    branch = cont.from_state({x: 0.5})
    np.testing.assert_allclose(branch.states[x], branch.parameter / 2, atol=1e-8)


def test_missing_parameter_values_are_reported():
    x, p, q = sympy.symbols('x p q')
    with pytest.raises(ValueError, match='q'):
        equilibrium_continuation({x: p - q * x}, {p: 1.0}, p)
