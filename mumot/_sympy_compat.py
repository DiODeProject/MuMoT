"""Wrappers keeping the SymPy behaviour MuMoT was written against."""

import sympy


def latex(expr, **settings) -> str:
    """Return the LaTeX form of ``expr``, as :func:`sympy.latex` does.

    Strings are assumed to already be LaTeX and are returned unchanged
    (as :func:`sympy.latex` did before SymPy 1.7; it now escapes them).
    """
    if isinstance(expr, str):
        return expr
    return sympy.latex(expr, **settings)


def simplify(expr, **kwargs):
    """:func:`sympy.simplify` without evaluating unevaluated objects.

    Since SymPy 1.6 ``simplify`` calls ``doit()`` by default, which e.g.
    evaluates ``Derivative(Phi_A, t)`` to zero when ``Phi_A`` is a plain
    symbol; MuMoT relies on such derivatives being kept.
    """
    kwargs.setdefault('doit', False)
    return sympy.simplify(expr, **kwargs)
